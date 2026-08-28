import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge50582
def owner : Owner := ⟨.program ⟨257⟩, ⟨65669⟩⟩
def mergeEvent : Nat := 50582
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50578RawTerms
def rightRaw : List Term := Proof.Events197.exact50548RawTerms
def group : MergeGroup := .operator 50578 50548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50578) (leftOrdinal := 1)
    (rightResult := 50548) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50582

namespace LeftMerge50590
def owner : Owner := ⟨.program ⟨257⟩, ⟨69329⟩⟩
def mergeEvent : Nat := 50590
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50584RawTerms
def rightRaw : List Term := Proof.Events197.exact50520RawTerms
def group : MergeGroup := .operator 50584 50520
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50584) (leftOrdinal := 1)
    (rightResult := 50520) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69328⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50590

namespace LeftMerge50592
def owner : Owner := ⟨.program ⟨257⟩, ⟨69329⟩⟩
def mergeEvent : Nat := 50592
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68578⟩⟩] } }
def rhsRaw : List Term := Proof.Events197.exact50517RawTerms
def group : MergeGroup := .relation 50591
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50591) (rhsResult := 50517)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69328⟩⟩) ⟨68578⟩ 50517) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68578⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], [⟨.program ⟨257⟩, ⟨68578⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50592

namespace LeftMerge50593
def owner : Owner := ⟨.program ⟨257⟩, ⟨69329⟩⟩
def mergeEvent : Nat := 50593
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50584RawTerms
def rightRaw : List Term := Proof.Events197.exact50520RawTerms
def group : MergeGroup := .operator 50584 50520
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50584) (leftOrdinal := 0)
    (rightResult := 50520) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69328⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50593

namespace LeftMerge50607
def owner : Owner := ⟨.program ⟨257⟩, ⟨67853⟩⟩
def mergeEvent : Nat := 50607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67850⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events197.exact50601RawTerms
def group : MergeGroup := .operator 46745 50601
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 50601) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67850⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67850⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50607

namespace LeftMerge50686
def owner : Owner := ⟨.program ⟨257⟩, ⟨65662⟩⟩
def mergeEvent : Nat := 50686
def frameStart : Nat := 50656
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events197.exact50682RawTerms
def rightRaw : List Term := Proof.Events197.exact50679RawTerms
def group : MergeGroup := .operator 50682 50679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50682) (leftOrdinal := 0)
    (rightResult := 50679) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50686

namespace LeftMerge50716
def owner : Owner := ⟨.program ⟨257⟩, ⟨68961⟩⟩
def mergeEvent : Nat := 50716
def frameStart : Nat := 50656
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50712RawTerms
def rightRaw : List Term := Proof.Events198.exact50710RawTerms
def group : MergeGroup := .operator 50712 50710
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50712) (leftOrdinal := 0)
    (rightResult := 50710) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50716

namespace LeftMerge50739
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def mergeEvent : Nat := 50739
def frameStart : Nat := 50656
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50735RawTerms
def rightRaw : List Term := Proof.Events198.exact50732RawTerms
def group : MergeGroup := .operator 50735 50732
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50735) (leftOrdinal := 0)
    (rightResult := 50732) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50739

namespace LeftMerge50748
def owner : Owner := ⟨.program ⟨257⟩, ⟨69331⟩⟩
def mergeEvent : Nat := 50748
def frameStart : Nat := 50656
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50744RawTerms
def rightRaw : List Term := Proof.Events198.exact50701RawTerms
def group : MergeGroup := .operator 50744 50701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50744) (leftOrdinal := 0)
    (rightResult := 50701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69328⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50748

namespace LeftMerge50749
def owner : Owner := ⟨.program ⟨257⟩, ⟨69331⟩⟩
def mergeEvent : Nat := 50749
def frameStart : Nat := 50656
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50744RawTerms
def rightRaw : List Term := Proof.Events198.exact50701RawTerms
def group : MergeGroup := .operator 50744 50701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50744) (leftOrdinal := 1)
    (rightResult := 50701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69328⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50749

namespace LeftMerge50751
def owner : Owner := ⟨.program ⟨257⟩, ⟨69331⟩⟩
def mergeEvent : Nat := 50751
def frameStart : Nat := 50656
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68578⟩⟩] } }
def rhsRaw : List Term := Proof.Events198.exact50698RawTerms
def group : MergeGroup := .relation 50750
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50750) (rhsResult := 50698)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69328⟩⟩) ⟨68578⟩ 50698) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68578⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], [⟨.program ⟨257⟩, ⟨68578⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50751

namespace LeftMerge50759
def owner : Owner := ⟨.program ⟨257⟩, ⟨65854⟩⟩
def mergeEvent : Nat := 50759
def frameStart : Nat := 50656
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50712RawTerms
def rightRaw : List Term := Proof.Events198.exact50755RawTerms
def group : MergeGroup := .operator 50712 50755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50712) (leftOrdinal := 0)
    (rightResult := 50755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65852⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50759

namespace LeftMerge50776
def owner : Owner := ⟨.program ⟨257⟩, ⟨67853⟩⟩
def mergeEvent : Nat := 50776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events198.exact50773RawTerms
def group : MergeGroup := .relation 50775
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50775) (rhsResult := 50773)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67850⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50774 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67850⟩⟩]⟩) (none) 50773) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50776

namespace LeftMerge50777
def owner : Owner := ⟨.program ⟨257⟩, ⟨67853⟩⟩
def mergeEvent : Nat := 50777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩] } }
def rhsRaw : List Term := Proof.Events198.exact50773RawTerms
def group : MergeGroup := .relation 50775
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50775) (rhsResult := 50773)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67850⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50774 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67850⟩⟩]⟩) (none) 50773) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69328⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50777

namespace LeftMerge50778
def owner : Owner := ⟨.program ⟨257⟩, ⟨67853⟩⟩
def mergeEvent : Nat := 50778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68578⟩⟩] } }
def rhsRaw : List Term := Proof.Events198.exact50773RawTerms
def group : MergeGroup := .relation 50775
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50775) (rhsResult := 50773)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67850⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50774 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67850⟩⟩]⟩) (none) 50773) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68578⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25826⟩⟩, ⟨.program ⟨257⟩, ⟨65661⟩⟩], [⟨.program ⟨257⟩, ⟨68578⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50778

namespace LeftMerge50779
def owner : Owner := ⟨.program ⟨257⟩, ⟨67853⟩⟩
def mergeEvent : Nat := 50779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨65852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events198.exact50773RawTerms
def group : MergeGroup := .relation 50775
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50775) (rhsResult := 50773)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67850⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50774 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67850⟩⟩]⟩) (none) 50773) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨65852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50779

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
