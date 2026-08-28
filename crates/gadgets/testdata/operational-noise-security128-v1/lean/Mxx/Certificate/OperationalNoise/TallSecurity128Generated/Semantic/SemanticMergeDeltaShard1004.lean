import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge163892
def owner : Owner := ⟨.program ⟨257⟩, ⟨49706⟩⟩
def mergeEvent : Nat := 163892
def frameStart : Nat := 163800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact163888RawTerms
def rightRaw : List Term := Proof.Events640.exact163845RawTerms
def group : MergeGroup := .operator 163888 163845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163888) (leftOrdinal := 0)
    (rightResult := 163845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49703⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163892

namespace LeftMerge163893
def owner : Owner := ⟨.program ⟨257⟩, ⟨49706⟩⟩
def mergeEvent : Nat := 163893
def frameStart : Nat := 163800
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact163888RawTerms
def rightRaw : List Term := Proof.Events640.exact163845RawTerms
def group : MergeGroup := .operator 163888 163845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163888) (leftOrdinal := 1)
    (rightResult := 163845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49703⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge163893

namespace LeftMerge163895
def owner : Owner := ⟨.program ⟨257⟩, ⟨49706⟩⟩
def mergeEvent : Nat := 163895
def frameStart : Nat := 163800
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49173⟩⟩] } }
def rhsRaw : List Term := Proof.Events640.exact163842RawTerms
def group : MergeGroup := .relation 163894
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 163894) (rhsResult := 163842)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49703⟩⟩) ⟨49173⟩ 163842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49173⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], [⟨.program ⟨257⟩, ⟨49173⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge163895

namespace LeftMerge163903
def owner : Owner := ⟨.program ⟨257⟩, ⟨48182⟩⟩
def mergeEvent : Nat := 163903
def frameStart : Nat := 163800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact163856RawTerms
def rightRaw : List Term := Proof.Events640.exact163899RawTerms
def group : MergeGroup := .operator 163856 163899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163856) (leftOrdinal := 0)
    (rightResult := 163899) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163903

namespace LeftMerge163920
def owner : Owner := ⟨.program ⟨257⟩, ⟨48632⟩⟩
def mergeEvent : Nat := 163920
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events640.exact163917RawTerms
def group : MergeGroup := .relation 163919
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 163919) (rhsResult := 163917)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48629⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 163918 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48629⟩⟩]⟩) (none) 163917) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163920

namespace LeftMerge163921
def owner : Owner := ⟨.program ⟨257⟩, ⟨48632⟩⟩
def mergeEvent : Nat := 163921
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩] } }
def rhsRaw : List Term := Proof.Events640.exact163917RawTerms
def group : MergeGroup := .relation 163919
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 163919) (rhsResult := 163917)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48629⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 163918 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48629⟩⟩]⟩) (none) 163917) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge163921

namespace LeftMerge163922
def owner : Owner := ⟨.program ⟨257⟩, ⟨48632⟩⟩
def mergeEvent : Nat := 163922
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49173⟩⟩] } }
def rhsRaw : List Term := Proof.Events640.exact163917RawTerms
def group : MergeGroup := .relation 163919
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 163919) (rhsResult := 163917)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48629⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 163918 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48629⟩⟩]⟩) (none) 163917) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49173⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], [⟨.program ⟨257⟩, ⟨49173⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163922

namespace LeftMerge163923
def owner : Owner := ⟨.program ⟨257⟩, ⟨48632⟩⟩
def mergeEvent : Nat := 163923
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events640.exact163917RawTerms
def group : MergeGroup := .relation 163919
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 163919) (rhsResult := 163917)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48629⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 163918 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48629⟩⟩]⟩) (none) 163917) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge163923

namespace LeftMerge163928
def owner : Owner := ⟨.program ⟨257⟩, ⟨49705⟩⟩
def mergeEvent : Nat := 163928
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49173⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact163924RawTerms
def rightRaw : List Term := Proof.Events639.exact163727RawTerms
def group : MergeGroup := .operator 163924 163727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163924) (leftOrdinal := 2)
    (rightResult := 163727) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49173⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49173⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], [⟨.program ⟨257⟩, ⟨49173⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge163928

namespace LeftMerge163929
def owner : Owner := ⟨.program ⟨257⟩, ⟨49705⟩⟩
def mergeEvent : Nat := 163929
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact163924RawTerms
def rightRaw : List Term := Proof.Events639.exact163727RawTerms
def group : MergeGroup := .operator 163924 163727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163924) (leftOrdinal := 1)
    (rightResult := 163727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163929

namespace LeftMerge163937
def owner : Owner := ⟨.program ⟨257⟩, ⟨50131⟩⟩
def mergeEvent : Nat := 163937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact163931RawTerms
def rightRaw : List Term := Proof.Events639.exact163638RawTerms
def group : MergeGroup := .operator 163931 163638
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163931) (leftOrdinal := 0)
    (rightResult := 163638) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50129⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163937

namespace LeftMerge163938
def owner : Owner := ⟨.program ⟨257⟩, ⟨50131⟩⟩
def mergeEvent : Nat := 163938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact163931RawTerms
def rightRaw : List Term := Proof.Events639.exact163638RawTerms
def group : MergeGroup := .operator 163931 163638
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163931) (leftOrdinal := 1)
    (rightResult := 163638) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50129⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge163938

namespace LeftMerge163940
def owner : Owner := ⟨.program ⟨257⟩, ⟨50131⟩⟩
def mergeEvent : Nat := 163940
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49337⟩⟩] } }
def rhsRaw : List Term := Proof.Events639.exact163635RawTerms
def group : MergeGroup := .relation 163939
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 163939) (rhsResult := 163635)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50129⟩⟩) ⟨49337⟩ 163635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49337⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨49337⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge163940

namespace LeftMerge163954
def owner : Owner := ⟨.program ⟨257⟩, ⟨48979⟩⟩
def mergeEvent : Nat := 163954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48976⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events640.exact163948RawTerms
def group : MergeGroup := .operator 163745 163948
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 163948) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48976⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48976⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge163954

namespace LeftMerge164075
def owner : Owner := ⟨.program ⟨257⟩, ⟨49524⟩⟩
def mergeEvent : Nat := 164075
def frameStart : Nat := 164009
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact164071RawTerms
def rightRaw : List Term := Proof.Events640.exact164069RawTerms
def group : MergeGroup := .operator 164071 164069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164071) (leftOrdinal := 0)
    (rightResult := 164069) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164075

namespace LeftMerge164087
def owner : Owner := ⟨.program ⟨257⟩, ⟨50130⟩⟩
def mergeEvent : Nat := 164087
def frameStart : Nat := 164009
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact164083RawTerms
def rightRaw : List Term := Proof.Events640.exact164060RawTerms
def group : MergeGroup := .operator 164083 164060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164083) (leftOrdinal := 0)
    (rightResult := 164060) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50129⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164087

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
