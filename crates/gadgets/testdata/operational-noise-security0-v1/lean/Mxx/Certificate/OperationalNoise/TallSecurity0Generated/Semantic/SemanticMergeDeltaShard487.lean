import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge79658
def owner : Owner := ⟨.program ⟨214⟩, ⟨30106⟩⟩
def mergeEvent : Nat := 79658
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def rhsRaw : List Term := Proof.Events023.exact6114RawTerms
def group : MergeGroup := .relation 79657
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79657) (rhsResult := 6114)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7827⟩⟩) ⟨6754⟩ 6114) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79658

namespace LeftMerge79659
def owner : Owner := ⟨.program ⟨214⟩, ⟨30106⟩⟩
def mergeEvent : Nat := 79659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩] } }
def leftRaw : List Term := Proof.Events310.exact79599RawTerms
def rightRaw : List Term := Proof.Events023.exact6121RawTerms
def group : MergeGroup := .operator 79599 6121
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79599) (leftOrdinal := 18)
    (rightResult := 6121) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7827⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79659

namespace LeftMerge79661
def owner : Owner := ⟨.program ⟨214⟩, ⟨30106⟩⟩
def mergeEvent : Nat := 79661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def rhsRaw : List Term := Proof.Events023.exact6114RawTerms
def group : MergeGroup := .relation 79660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79660) (rhsResult := 6114)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7827⟩⟩) ⟨6754⟩ 6114) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79661

namespace LeftMerge79662
def owner : Owner := ⟨.program ⟨214⟩, ⟨30106⟩⟩
def mergeEvent : Nat := 79662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩] } }
def leftRaw : List Term := Proof.Events310.exact79599RawTerms
def rightRaw : List Term := Proof.Events023.exact6121RawTerms
def group : MergeGroup := .operator 79599 6121
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79599) (leftOrdinal := 0)
    (rightResult := 6121) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7827⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79662

namespace LeftMerge79667
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79667
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 6)
    (rightResult := 65260) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79667

namespace LeftMerge79668
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79668
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 8)
    (rightResult := 65260) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79668

namespace LeftMerge79669
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 9)
    (rightResult := 65260) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16923⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79669

namespace LeftMerge79670
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 10)
    (rightResult := 65260) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79670

namespace LeftMerge79671
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 12)
    (rightResult := 65260) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79671

namespace LeftMerge79672
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79672
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 13)
    (rightResult := 65260) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79672

namespace LeftMerge79673
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79673
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 14)
    (rightResult := 65260) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79673

namespace LeftMerge79674
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79674
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 16)
    (rightResult := 65260) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79674

namespace LeftMerge79675
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79675
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 17)
    (rightResult := 65260) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79675

namespace LeftMerge79676
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79676
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 19)
    (rightResult := 65260) (rightOrdinal := 37) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79676

namespace LeftMerge79677
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79677
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 1)
    (rightResult := 65260) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79677

namespace LeftMerge79678
def owner : Owner := ⟨.program ⟨214⟩, ⟨30107⟩⟩
def mergeEvent : Nat := 79678
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79663RawTerms
def rightRaw : List Term := Proof.Events254.exact65260RawTerms
def group : MergeGroup := .operator 79663 65260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79663) (leftOrdinal := 2)
    (rightResult := 65260) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6754⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79678

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
