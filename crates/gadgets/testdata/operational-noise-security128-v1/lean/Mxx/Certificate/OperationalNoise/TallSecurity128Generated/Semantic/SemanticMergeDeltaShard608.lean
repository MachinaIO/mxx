import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge101654
def owner : Owner := ⟨.program ⟨257⟩, ⟨40955⟩⟩
def mergeEvent : Nat := 101654
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41305⟩⟩] } }
def rhsRaw : List Term := Proof.Events397.exact101649RawTerms
def group : MergeGroup := .relation 101651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101651) (rhsResult := 101649)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40952⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101650 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40952⟩⟩]⟩) (none) 101649) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41305⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨41305⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101654

namespace LeftMerge101655
def owner : Owner := ⟨.program ⟨257⟩, ⟨40955⟩⟩
def mergeEvent : Nat := 101655
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events397.exact101649RawTerms
def group : MergeGroup := .relation 101651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101651) (rhsResult := 101649)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40952⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101650 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40952⟩⟩]⟩) (none) 101649) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40387⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101655

namespace LeftMerge101660
def owner : Owner := ⟨.program ⟨257⟩, ⟨42111⟩⟩
def mergeEvent : Nat := 101660
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42108⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101656RawTerms
def rightRaw : List Term := Proof.Events396.exact101478RawTerms
def group : MergeGroup := .operator 101656 101478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101656) (leftOrdinal := 0)
    (rightResult := 101478) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42108⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42108⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42108⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101660

namespace LeftMerge101661
def owner : Owner := ⟨.program ⟨257⟩, ⟨42111⟩⟩
def mergeEvent : Nat := 101661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41305⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101656RawTerms
def rightRaw : List Term := Proof.Events396.exact101478RawTerms
def group : MergeGroup := .operator 101656 101478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101656) (leftOrdinal := 2)
    (rightResult := 101478) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41305⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41305⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨41305⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101661

namespace LeftMerge101669
def owner : Owner := ⟨.program ⟨257⟩, ⟨42112⟩⟩
def mergeEvent : Nat := 101669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101663RawTerms
def rightRaw : List Term := Proof.Events060.exact15602RawTerms
def group : MergeGroup := .operator 101663 15602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101663) (leftOrdinal := 0)
    (rightResult := 15602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101669

namespace LeftMerge101670
def owner : Owner := ⟨.program ⟨257⟩, ⟨42112⟩⟩
def mergeEvent : Nat := 101670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101663RawTerms
def rightRaw : List Term := Proof.Events060.exact15602RawTerms
def group : MergeGroup := .operator 101663 15602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101663) (leftOrdinal := 1)
    (rightResult := 15602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101670

namespace LeftMerge101672
def owner : Owner := ⟨.program ⟨257⟩, ⟨42112⟩⟩
def mergeEvent : Nat := 101672
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15595RawTerms
def group : MergeGroup := .relation 101671
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101671) (rhsResult := 15595)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7159⟩⟩) ⟨7045⟩ 15595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101672

namespace LeftMerge101686
def owner : Owner := ⟨.program ⟨257⟩, ⟨39430⟩⟩
def mergeEvent : Nat := 101686
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39428⟩⟩] } }
def leftRaw : List Term := Proof.Events362.exact92734RawTerms
def rightRaw : List Term := Proof.Events397.exact101680RawTerms
def group : MergeGroup := .operator 92734 101680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92734) (leftOrdinal := 0)
    (rightResult := 101680) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39428⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39428⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101686

namespace LeftMerge101687
def owner : Owner := ⟨.program ⟨257⟩, ⟨39430⟩⟩
def mergeEvent : Nat := 101687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39428⟩⟩] } }
def leftRaw : List Term := Proof.Events362.exact92734RawTerms
def rightRaw : List Term := Proof.Events397.exact101680RawTerms
def group : MergeGroup := .operator 92734 101680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92734) (leftOrdinal := 1)
    (rightResult := 101680) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39428⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37468⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39428⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101687

namespace LeftMerge101689
def owner : Owner := ⟨.program ⟨257⟩, ⟨39430⟩⟩
def mergeEvent : Nat := 101689
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38625⟩⟩] } }
def rhsRaw : List Term := Proof.Events397.exact101677RawTerms
def group : MergeGroup := .relation 101688
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101688) (rhsResult := 101677)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37468⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39428⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39428⟩⟩) ⟨38625⟩ 101677) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38625⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37468⟩⟩], [⟨.program ⟨257⟩, ⟨38625⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101689

namespace LeftMerge101703
def owner : Owner := ⟨.program ⟨257⟩, ⟨38275⟩⟩
def mergeEvent : Nat := 101703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38272⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events397.exact101697RawTerms
def group : MergeGroup := .operator 90620 101697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 101697) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38272⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38272⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101703

namespace LeftMerge101824
def owner : Owner := ⟨.program ⟨257⟩, ⟨38808⟩⟩
def mergeEvent : Nat := 101824
def frameStart : Nat := 101758
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101820RawTerms
def rightRaw : List Term := Proof.Events397.exact101818RawTerms
def group : MergeGroup := .operator 101820 101818
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101820) (leftOrdinal := 0)
    (rightResult := 101818) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37468⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101824

namespace LeftMerge101836
def owner : Owner := ⟨.program ⟨257⟩, ⟨39429⟩⟩
def mergeEvent : Nat := 101836
def frameStart : Nat := 101758
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39428⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101832RawTerms
def rightRaw : List Term := Proof.Events397.exact101809RawTerms
def group : MergeGroup := .operator 101832 101809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101832) (leftOrdinal := 0)
    (rightResult := 101809) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39428⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39428⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101836

namespace LeftMerge101837
def owner : Owner := ⟨.program ⟨257⟩, ⟨39429⟩⟩
def mergeEvent : Nat := 101837
def frameStart : Nat := 101758
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39428⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101832RawTerms
def rightRaw : List Term := Proof.Events397.exact101809RawTerms
def group : MergeGroup := .operator 101832 101809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101832) (leftOrdinal := 1)
    (rightResult := 101809) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39428⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37468⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39428⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101837

namespace LeftMerge101839
def owner : Owner := ⟨.program ⟨257⟩, ⟨39429⟩⟩
def mergeEvent : Nat := 101839
def frameStart : Nat := 101758
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38625⟩⟩] } }
def rhsRaw : List Term := Proof.Events397.exact101806RawTerms
def group : MergeGroup := .relation 101838
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101838) (rhsResult := 101806)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37468⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39428⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39428⟩⟩) ⟨38625⟩ 101806) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38625⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37468⟩⟩], [⟨.program ⟨257⟩, ⟨38625⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101839

namespace LeftMerge101847
def owner : Owner := ⟨.program ⟨257⟩, ⟨37706⟩⟩
def mergeEvent : Nat := 101847
def frameStart : Nat := 101758
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37704⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101820RawTerms
def rightRaw : List Term := Proof.Events397.exact101843RawTerms
def group : MergeGroup := .operator 101820 101843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101820) (leftOrdinal := 0)
    (rightResult := 101843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37704⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101847

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
