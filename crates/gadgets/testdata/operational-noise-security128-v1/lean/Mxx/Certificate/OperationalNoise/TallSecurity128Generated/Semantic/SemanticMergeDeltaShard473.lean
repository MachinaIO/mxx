import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge79748
def owner : Owner := ⟨.program ⟨257⟩, ⟨28442⟩⟩
def mergeEvent : Nat := 79748
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27615⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79743RawTerms
def rightRaw : List Term := Proof.Events310.exact79565RawTerms
def group : MergeGroup := .operator 79743 79565
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79743) (leftOrdinal := 2)
    (rightResult := 79565) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27615⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27615⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79748

namespace LeftMerge79774
def owner : Owner := ⟨.program ⟨257⟩, ⟨25803⟩⟩
def mergeEvent : Nat := 79774
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events012.exact3270RawTerms
def rightRaw : List Term := Proof.Events296.exact75903RawTerms
def group : MergeGroup := .operator 3270 75903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3270) (leftOrdinal := 0)
    (rightResult := 75903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25802⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79774

namespace LeftMerge79779
def owner : Owner := ⟨.program ⟨257⟩, ⟨10334⟩⟩
def mergeEvent : Nat := 79779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75773RawTerms
def rightRaw : List Term := Proof.Events082.exact21088RawTerms
def group : MergeGroup := .operator 75773 21088
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75773) (leftOrdinal := 0)
    (rightResult := 21088) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79779

namespace LeftMerge79796
def owner : Owner := ⟨.program ⟨257⟩, ⟨65610⟩⟩
def mergeEvent : Nat := 79796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79790RawTerms
def rightRaw : List Term := Proof.Events012.exact3273RawTerms
def group : MergeGroup := .operator 79790 3273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79790) (leftOrdinal := 1)
    (rightResult := 3273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79796

namespace LeftMerge79797
def owner : Owner := ⟨.program ⟨257⟩, ⟨65610⟩⟩
def mergeEvent : Nat := 79797
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79790RawTerms
def rightRaw : List Term := Proof.Events012.exact3273RawTerms
def group : MergeGroup := .operator 79790 3273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79790) (leftOrdinal := 0)
    (rightResult := 3273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79797

namespace LeftMerge79802
def owner : Owner := ⟨.program ⟨257⟩, ⟨65611⟩⟩
def mergeEvent : Nat := 79802
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events012.exact3273RawTerms
def rightRaw : List Term := Proof.Events296.exact75903RawTerms
def group : MergeGroup := .operator 3273 75903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3273) (leftOrdinal := 0)
    (rightResult := 75903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79802

namespace LeftMerge79807
def owner : Owner := ⟨.program ⟨257⟩, ⟨10352⟩⟩
def mergeEvent : Nat := 79807
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75773RawTerms
def rightRaw : List Term := Proof.Events082.exact21129RawTerms
def group : MergeGroup := .operator 75773 21129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75773) (leftOrdinal := 0)
    (rightResult := 21129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79807

namespace LeftMerge79824
def owner : Owner := ⟨.program ⟨257⟩, ⟨65614⟩⟩
def mergeEvent : Nat := 79824
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79818RawTerms
def rightRaw : List Term := Proof.Events082.exact21118RawTerms
def group : MergeGroup := .operator 79818 21118
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79818) (leftOrdinal := 1)
    (rightResult := 21118) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79824

namespace LeftMerge79826
def owner : Owner := ⟨.program ⟨257⟩, ⟨65614⟩⟩
def mergeEvent : Nat := 79826
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def rhsRaw : List Term := Proof.Events082.exact21088RawTerms
def group : MergeGroup := .relation 79825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79825) (rhsResult := 21088)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9541⟩⟩) ⟨7276⟩ 21088) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79826

namespace LeftMerge79827
def owner : Owner := ⟨.program ⟨257⟩, ⟨65614⟩⟩
def mergeEvent : Nat := 79827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79818RawTerms
def rightRaw : List Term := Proof.Events082.exact21118RawTerms
def group : MergeGroup := .operator 79818 21118
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79818) (leftOrdinal := 0)
    (rightResult := 21118) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79827

namespace LeftMerge79832
def owner : Owner := ⟨.program ⟨257⟩, ⟨65615⟩⟩
def mergeEvent : Nat := 79832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79828RawTerms
def rightRaw : List Term := Proof.Events311.exact79798RawTerms
def group : MergeGroup := .operator 79828 79798
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79828) (leftOrdinal := 1)
    (rightResult := 79798) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79832

namespace LeftMerge79840
def owner : Owner := ⟨.program ⟨257⟩, ⟨69307⟩⟩
def mergeEvent : Nat := 79840
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79834RawTerms
def rightRaw : List Term := Proof.Events311.exact79770RawTerms
def group : MergeGroup := .operator 79834 79770
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79834) (leftOrdinal := 1)
    (rightResult := 79770) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69306⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79840

namespace LeftMerge79842
def owner : Owner := ⟨.program ⟨257⟩, ⟨69307⟩⟩
def mergeEvent : Nat := 79842
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68566⟩⟩] } }
def rhsRaw : List Term := Proof.Events311.exact79767RawTerms
def group : MergeGroup := .relation 79841
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79841) (rhsResult := 79767)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69306⟩⟩) ⟨68566⟩ 79767) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68566⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79842

namespace LeftMerge79843
def owner : Owner := ⟨.program ⟨257⟩, ⟨69307⟩⟩
def mergeEvent : Nat := 79843
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79834RawTerms
def rightRaw : List Term := Proof.Events311.exact79770RawTerms
def group : MergeGroup := .operator 79834 79770
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79834) (leftOrdinal := 0)
    (rightResult := 79770) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69306⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79843

namespace LeftMerge79857
def owner : Owner := ⟨.program ⟨257⟩, ⟨67833⟩⟩
def mergeEvent : Nat := 79857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67830⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events311.exact79851RawTerms
def group : MergeGroup := .operator 75995 79851
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 79851) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67830⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79857

namespace LeftMerge79936
def owner : Owner := ⟨.program ⟨257⟩, ⟨65608⟩⟩
def mergeEvent : Nat := 79936
def frameStart : Nat := 79906
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events312.exact79932RawTerms
def rightRaw : List Term := Proof.Events312.exact79929RawTerms
def group : MergeGroup := .operator 79932 79929
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79932) (leftOrdinal := 0)
    (rightResult := 79929) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65607⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25802⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79936

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
