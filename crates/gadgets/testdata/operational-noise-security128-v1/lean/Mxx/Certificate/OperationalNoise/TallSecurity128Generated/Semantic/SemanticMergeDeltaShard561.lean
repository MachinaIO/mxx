import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge93688
def owner : Owner := ⟨.program ⟨257⟩, ⟨29582⟩⟩
def mergeEvent : Nat := 93688
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30654⟩⟩] } }
def rhsRaw : List Term := Proof.Events365.exact93684RawTerms
def group : MergeGroup := .relation 93686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93686) (rhsResult := 93684)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29579⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93685 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29579⟩⟩]⟩) (none) 93684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30654⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30654⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93688

namespace LeftMerge93689
def owner : Owner := ⟨.program ⟨257⟩, ⟨29582⟩⟩
def mergeEvent : Nat := 93689
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30119⟩⟩] } }
def rhsRaw : List Term := Proof.Events365.exact93684RawTerms
def group : MergeGroup := .relation 93686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93686) (rhsResult := 93684)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29579⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93685 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29579⟩⟩]⟩) (none) 93684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30119⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], [⟨.program ⟨257⟩, ⟨30119⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93689

namespace LeftMerge93690
def owner : Owner := ⟨.program ⟨257⟩, ⟨29582⟩⟩
def mergeEvent : Nat := 93690
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events365.exact93684RawTerms
def group : MergeGroup := .relation 93686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93686) (rhsResult := 93684)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29579⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93685 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29579⟩⟩]⟩) (none) 93684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29128⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93690

namespace LeftMerge93695
def owner : Owner := ⟨.program ⟨257⟩, ⟨30656⟩⟩
def mergeEvent : Nat := 93695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30119⟩⟩] } }
def leftRaw : List Term := Proof.Events365.exact93691RawTerms
def rightRaw : List Term := Proof.Events365.exact93505RawTerms
def group : MergeGroup := .operator 93691 93505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93691) (leftOrdinal := 2)
    (rightResult := 93505) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30119⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30119⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], [⟨.program ⟨257⟩, ⟨30119⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93695

namespace LeftMerge93696
def owner : Owner := ⟨.program ⟨257⟩, ⟨30656⟩⟩
def mergeEvent : Nat := 93696
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30654⟩⟩] } }
def leftRaw : List Term := Proof.Events365.exact93691RawTerms
def rightRaw : List Term := Proof.Events365.exact93505RawTerms
def group : MergeGroup := .operator 93691 93505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93691) (leftOrdinal := 1)
    (rightResult := 93505) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30654⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30654⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30654⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93696

namespace LeftMerge93704
def owner : Owner := ⟨.program ⟨257⟩, ⟨31096⟩⟩
def mergeEvent : Nat := 93704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩] } }
def leftRaw : List Term := Proof.Events366.exact93698RawTerms
def rightRaw : List Term := Proof.Events364.exact93421RawTerms
def group : MergeGroup := .operator 93698 93421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93698) (leftOrdinal := 0)
    (rightResult := 93421) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31094⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93704

namespace LeftMerge93705
def owner : Owner := ⟨.program ⟨257⟩, ⟨31096⟩⟩
def mergeEvent : Nat := 93705
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩] } }
def leftRaw : List Term := Proof.Events366.exact93698RawTerms
def rightRaw : List Term := Proof.Events364.exact93421RawTerms
def group : MergeGroup := .operator 93698 93421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93698) (leftOrdinal := 1)
    (rightResult := 93421) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31094⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93705

namespace LeftMerge93707
def owner : Owner := ⟨.program ⟨257⟩, ⟨31096⟩⟩
def mergeEvent : Nat := 93707
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30286⟩⟩] } }
def rhsRaw : List Term := Proof.Events364.exact93418RawTerms
def group : MergeGroup := .relation 93706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93706) (rhsResult := 93418)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31094⟩⟩) ⟨30286⟩ 93418) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30286⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93707

namespace LeftMerge93721
def owner : Owner := ⟨.program ⟨257⟩, ⟨29939⟩⟩
def mergeEvent : Nat := 93721
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events366.exact93715RawTerms
def group : MergeGroup := .operator 90620 93715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 93715) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29936⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93721

namespace LeftMerge93842
def owner : Owner := ⟨.program ⟨257⟩, ⟨30468⟩⟩
def mergeEvent : Nat := 93842
def frameStart : Nat := 93776
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29128⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events366.exact93838RawTerms
def rightRaw : List Term := Proof.Events366.exact93836RawTerms
def group : MergeGroup := .operator 93838 93836
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93838) (leftOrdinal := 0)
    (rightResult := 93836) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29128⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93842

namespace LeftMerge93854
def owner : Owner := ⟨.program ⟨257⟩, ⟨31095⟩⟩
def mergeEvent : Nat := 93854
def frameStart : Nat := 93776
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩] } }
def leftRaw : List Term := Proof.Events366.exact93850RawTerms
def rightRaw : List Term := Proof.Events366.exact93827RawTerms
def group : MergeGroup := .operator 93850 93827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93850) (leftOrdinal := 0)
    (rightResult := 93827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31094⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93854

namespace LeftMerge93855
def owner : Owner := ⟨.program ⟨257⟩, ⟨31095⟩⟩
def mergeEvent : Nat := 93855
def frameStart : Nat := 93776
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29128⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩] } }
def leftRaw : List Term := Proof.Events366.exact93850RawTerms
def rightRaw : List Term := Proof.Events366.exact93827RawTerms
def group : MergeGroup := .operator 93850 93827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93850) (leftOrdinal := 1)
    (rightResult := 93827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29128⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31094⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93855

namespace LeftMerge93857
def owner : Owner := ⟨.program ⟨257⟩, ⟨31095⟩⟩
def mergeEvent : Nat := 93857
def frameStart : Nat := 93776
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29128⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30286⟩⟩] } }
def rhsRaw : List Term := Proof.Events366.exact93824RawTerms
def group : MergeGroup := .relation 93856
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93856) (rhsResult := 93824)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31094⟩⟩) ⟨30286⟩ 93824) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30286⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93857

namespace LeftMerge93865
def owner : Owner := ⟨.program ⟨257⟩, ⟨29365⟩⟩
def mergeEvent : Nat := 93865
def frameStart : Nat := 93776
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29364⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events366.exact93838RawTerms
def rightRaw : List Term := Proof.Events366.exact93861RawTerms
def group : MergeGroup := .operator 93838 93861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 93838) (leftOrdinal := 0)
    (rightResult := 93861) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29364⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93865

namespace LeftMerge93882
def owner : Owner := ⟨.program ⟨257⟩, ⟨29939⟩⟩
def mergeEvent : Nat := 93882
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }
def rhsRaw : List Term := Proof.Events366.exact93879RawTerms
def group : MergeGroup := .relation 93881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93881) (rhsResult := 93879)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93880 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩) (none) 93879) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge93882

namespace LeftMerge93883
def owner : Owner := ⟨.program ⟨257⟩, ⟨29939⟩⟩
def mergeEvent : Nat := 93883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩] } }
def rhsRaw : List Term := Proof.Events366.exact93879RawTerms
def group : MergeGroup := .relation 93881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 93881) (rhsResult := 93879)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 93880 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩) (none) 93879) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge93883

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
