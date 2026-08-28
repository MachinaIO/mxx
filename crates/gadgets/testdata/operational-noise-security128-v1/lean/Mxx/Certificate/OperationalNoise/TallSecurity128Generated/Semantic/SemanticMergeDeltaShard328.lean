import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge56934
def owner : Owner := ⟨.program ⟨257⟩, ⟨71504⟩⟩
def mergeEvent : Nat := 56934
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def leftRaw : List Term := Proof.Events222.exact56899RawTerms
def rightRaw : List Term := Proof.Events216.exact55483RawTerms
def group : MergeGroup := .operator 56899 55483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56899) (leftOrdinal := 21)
    (rightResult := 55483) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56934

namespace LeftMerge56935
def owner : Owner := ⟨.program ⟨257⟩, ⟨71504⟩⟩
def mergeEvent : Nat := 56935
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events222.exact56899RawTerms
def rightRaw : List Term := Proof.Events216.exact55483RawTerms
def group : MergeGroup := .operator 56899 55483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56899) (leftOrdinal := 1)
    (rightResult := 55483) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56935

namespace LeftMerge56936
def owner : Owner := ⟨.program ⟨257⟩, ⟨71504⟩⟩
def mergeEvent : Nat := 56936
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def leftRaw : List Term := Proof.Events222.exact56899RawTerms
def rightRaw : List Term := Proof.Events216.exact55483RawTerms
def group : MergeGroup := .operator 56899 55483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56899) (leftOrdinal := 20)
    (rightResult := 55483) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56936

namespace LeftMerge56937
def owner : Owner := ⟨.program ⟨257⟩, ⟨71504⟩⟩
def mergeEvent : Nat := 56937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events222.exact56899RawTerms
def rightRaw : List Term := Proof.Events216.exact55483RawTerms
def group : MergeGroup := .operator 56899 55483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56899) (leftOrdinal := 0)
    (rightResult := 55483) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56937

namespace LeftMerge56938
def owner : Owner := ⟨.program ⟨257⟩, ⟨71504⟩⟩
def mergeEvent : Nat := 56938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def leftRaw : List Term := Proof.Events222.exact56899RawTerms
def rightRaw : List Term := Proof.Events216.exact55483RawTerms
def group : MergeGroup := .operator 56899 55483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56899) (leftOrdinal := 19)
    (rightResult := 55483) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56938

namespace LeftMerge56946
def owner : Owner := ⟨.program ⟨257⟩, ⟨71505⟩⟩
def mergeEvent : Nat := 56946
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events222.exact56940RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 56940 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56940) (leftOrdinal := 0)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56946

namespace LeftMerge56947
def owner : Owner := ⟨.program ⟨257⟩, ⟨71505⟩⟩
def mergeEvent : Nat := 56947
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events222.exact56940RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 56940 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56940) (leftOrdinal := 1)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56947

namespace LeftMerge56949
def owner : Owner := ⟨.program ⟨257⟩, ⟨71505⟩⟩
def mergeEvent : Nat := 56949
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15515RawTerms
def group : MergeGroup := .relation 56948
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56948) (rhsResult := 15515)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7139⟩⟩) ⟨7035⟩ 15515) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56949

namespace LeftMerge56963
def owner : Owner := ⟨.program ⟨257⟩, ⟨50225⟩⟩
def mergeEvent : Nat := 56963
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩] } }
def leftRaw : List Term := Proof.Events183.exact46931RawTerms
def rightRaw : List Term := Proof.Events222.exact56957RawTerms
def group : MergeGroup := .operator 46931 56957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46931) (leftOrdinal := 0)
    (rightResult := 56957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50223⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56963

namespace LeftMerge56964
def owner : Owner := ⟨.program ⟨257⟩, ⟨50225⟩⟩
def mergeEvent : Nat := 56964
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩] } }
def leftRaw : List Term := Proof.Events183.exact46931RawTerms
def rightRaw : List Term := Proof.Events222.exact56957RawTerms
def group : MergeGroup := .operator 46931 56957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46931) (leftOrdinal := 1)
    (rightResult := 56957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50223⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56964

namespace LeftMerge56966
def owner : Owner := ⟨.program ⟨257⟩, ⟨50225⟩⟩
def mergeEvent : Nat := 56966
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49372⟩⟩] } }
def rhsRaw : List Term := Proof.Events222.exact56954RawTerms
def group : MergeGroup := .relation 56965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56965) (rhsResult := 56954)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50223⟩⟩) ⟨49372⟩ 56954) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49372⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49372⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56966

namespace LeftMerge56980
def owner : Owner := ⟨.program ⟨257⟩, ⟨49055⟩⟩
def mergeEvent : Nat := 56980
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49052⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events222.exact56974RawTerms
def group : MergeGroup := .operator 46745 56974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 56974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49052⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49052⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56980

namespace LeftMerge57101
def owner : Owner := ⟨.program ⟨257⟩, ⟨49540⟩⟩
def mergeEvent : Nat := 57101
def frameStart : Nat := 57035
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57097RawTerms
def rightRaw : List Term := Proof.Events223.exact57095RawTerms
def group : MergeGroup := .operator 57097 57095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57097) (leftOrdinal := 0)
    (rightResult := 57095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57101

namespace LeftMerge57113
def owner : Owner := ⟨.program ⟨257⟩, ⟨50224⟩⟩
def mergeEvent : Nat := 57113
def frameStart : Nat := 57035
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57109RawTerms
def rightRaw : List Term := Proof.Events222.exact57086RawTerms
def group : MergeGroup := .operator 57109 57086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57109) (leftOrdinal := 0)
    (rightResult := 57086) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50223⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57113

namespace LeftMerge57114
def owner : Owner := ⟨.program ⟨257⟩, ⟨50224⟩⟩
def mergeEvent : Nat := 57114
def frameStart : Nat := 57035
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩] } }
def leftRaw : List Term := Proof.Events223.exact57109RawTerms
def rightRaw : List Term := Proof.Events222.exact57086RawTerms
def group : MergeGroup := .operator 57109 57086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57109) (leftOrdinal := 1)
    (rightResult := 57086) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50223⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57114

namespace LeftMerge57116
def owner : Owner := ⟨.program ⟨257⟩, ⟨50224⟩⟩
def mergeEvent : Nat := 57116
def frameStart : Nat := 57035
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49372⟩⟩] } }
def rhsRaw : List Term := Proof.Events222.exact57083RawTerms
def group : MergeGroup := .relation 57115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57115) (rhsResult := 57083)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50223⟩⟩) ⟨49372⟩ 57083) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49372⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49372⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57116

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
