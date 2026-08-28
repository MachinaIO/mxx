import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge91669
def owner : Owner := ⟨.program ⟨257⟩, ⟨42595⟩⟩
def mergeEvent : Nat := 91669
def frameStart : Nat := 91639
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events358.exact91665RawTerms
def rightRaw : List Term := Proof.Events358.exact91662RawTerms
def group : MergeGroup := .operator 91665 91662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91665) (leftOrdinal := 0)
    (rightResult := 91662) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14556⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91669

namespace LeftMerge91699
def owner : Owner := ⟨.program ⟨257⟩, ⟨44088⟩⟩
def mergeEvent : Nat := 91699
def frameStart : Nat := 91639
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events358.exact91695RawTerms
def rightRaw : List Term := Proof.Events358.exact91693RawTerms
def group : MergeGroup := .operator 91695 91693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91695) (leftOrdinal := 0)
    (rightResult := 91693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91699

namespace LeftMerge91722
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def mergeEvent : Nat := 91722
def frameStart : Nat := 91639
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events358.exact91718RawTerms
def rightRaw : List Term := Proof.Events358.exact91715RawTerms
def group : MergeGroup := .operator 91718 91715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91718) (leftOrdinal := 0)
    (rightResult := 91715) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91722

namespace LeftMerge91731
def owner : Owner := ⟨.program ⟨257⟩, ⟨44357⟩⟩
def mergeEvent : Nat := 91731
def frameStart : Nat := 91639
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩] } }
def leftRaw : List Term := Proof.Events358.exact91727RawTerms
def rightRaw : List Term := Proof.Events358.exact91684RawTerms
def group : MergeGroup := .operator 91727 91684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91727) (leftOrdinal := 0)
    (rightResult := 91684) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44354⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91731

namespace LeftMerge91732
def owner : Owner := ⟨.program ⟨257⟩, ⟨44357⟩⟩
def mergeEvent : Nat := 91732
def frameStart : Nat := 91639
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩] } }
def leftRaw : List Term := Proof.Events358.exact91727RawTerms
def rightRaw : List Term := Proof.Events358.exact91684RawTerms
def group : MergeGroup := .operator 91727 91684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91727) (leftOrdinal := 1)
    (rightResult := 91684) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44354⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91732

namespace LeftMerge91734
def owner : Owner := ⟨.program ⟨257⟩, ⟨44357⟩⟩
def mergeEvent : Nat := 91734
def frameStart : Nat := 91639
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43819⟩⟩] } }
def rhsRaw : List Term := Proof.Events358.exact91681RawTerms
def group : MergeGroup := .relation 91733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91733) (rhsResult := 91681)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44354⟩⟩) ⟨43819⟩ 91681) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43819⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], [⟨.program ⟨257⟩, ⟨43819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91734

namespace LeftMerge91742
def owner : Owner := ⟨.program ⟨257⟩, ⟨42830⟩⟩
def mergeEvent : Nat := 91742
def frameStart : Nat := 91639
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events358.exact91695RawTerms
def rightRaw : List Term := Proof.Events358.exact91738RawTerms
def group : MergeGroup := .operator 91695 91738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91695) (leftOrdinal := 0)
    (rightResult := 91738) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91742

namespace LeftMerge91759
def owner : Owner := ⟨.program ⟨257⟩, ⟨43282⟩⟩
def mergeEvent : Nat := 91759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events358.exact91756RawTerms
def group : MergeGroup := .relation 91758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91758) (rhsResult := 91756)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 91757 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43279⟩⟩]⟩) (none) 91756) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91759

namespace LeftMerge91760
def owner : Owner := ⟨.program ⟨257⟩, ⟨43282⟩⟩
def mergeEvent : Nat := 91760
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩] } }
def rhsRaw : List Term := Proof.Events358.exact91756RawTerms
def group : MergeGroup := .relation 91758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91758) (rhsResult := 91756)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 91757 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43279⟩⟩]⟩) (none) 91756) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91760

namespace LeftMerge91761
def owner : Owner := ⟨.program ⟨257⟩, ⟨43282⟩⟩
def mergeEvent : Nat := 91761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43819⟩⟩] } }
def rhsRaw : List Term := Proof.Events358.exact91756RawTerms
def group : MergeGroup := .relation 91758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91758) (rhsResult := 91756)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 91757 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43279⟩⟩]⟩) (none) 91756) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43819⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], [⟨.program ⟨257⟩, ⟨43819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91761

namespace LeftMerge91762
def owner : Owner := ⟨.program ⟨257⟩, ⟨43282⟩⟩
def mergeEvent : Nat := 91762
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨42828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events358.exact91756RawTerms
def group : MergeGroup := .relation 91758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91758) (rhsResult := 91756)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 91757 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43279⟩⟩]⟩) (none) 91756) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨42828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91762

namespace LeftMerge91767
def owner : Owner := ⟨.program ⟨257⟩, ⟨44356⟩⟩
def mergeEvent : Nat := 91767
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43819⟩⟩] } }
def leftRaw : List Term := Proof.Events358.exact91763RawTerms
def rightRaw : List Term := Proof.Events357.exact91577RawTerms
def group : MergeGroup := .operator 91763 91577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91763) (leftOrdinal := 2)
    (rightResult := 91577) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43819⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14556⟩⟩, ⟨.program ⟨257⟩, ⟨42594⟩⟩], [⟨.program ⟨257⟩, ⟨43819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91767

namespace LeftMerge91768
def owner : Owner := ⟨.program ⟨257⟩, ⟨44356⟩⟩
def mergeEvent : Nat := 91768
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩] } }
def leftRaw : List Term := Proof.Events358.exact91763RawTerms
def rightRaw : List Term := Proof.Events357.exact91577RawTerms
def group : MergeGroup := .operator 91763 91577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91763) (leftOrdinal := 1)
    (rightResult := 91577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44354⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91768

namespace LeftMerge91776
def owner : Owner := ⟨.program ⟨257⟩, ⟨44796⟩⟩
def mergeEvent : Nat := 91776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44794⟩⟩] } }
def leftRaw : List Term := Proof.Events358.exact91770RawTerms
def rightRaw : List Term := Proof.Events357.exact91493RawTerms
def group : MergeGroup := .operator 91770 91493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91770) (leftOrdinal := 0)
    (rightResult := 91493) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44794⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44794⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91776

namespace LeftMerge91777
def owner : Owner := ⟨.program ⟨257⟩, ⟨44796⟩⟩
def mergeEvent : Nat := 91777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨42828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44794⟩⟩] } }
def leftRaw : List Term := Proof.Events358.exact91770RawTerms
def rightRaw : List Term := Proof.Events357.exact91493RawTerms
def group : MergeGroup := .operator 91770 91493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91770) (leftOrdinal := 1)
    (rightResult := 91493) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨42828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44794⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨42828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91777

namespace LeftMerge91779
def owner : Owner := ⟨.program ⟨257⟩, ⟨44796⟩⟩
def mergeEvent : Nat := 91779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨42828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43986⟩⟩] } }
def rhsRaw : List Term := Proof.Events357.exact91490RawTerms
def group : MergeGroup := .relation 91778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91778) (rhsResult := 91490)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨42828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44794⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44794⟩⟩) ⟨43986⟩ 91490) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43986⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨42828⟩⟩], [⟨.program ⟨257⟩, ⟨43986⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91779

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
