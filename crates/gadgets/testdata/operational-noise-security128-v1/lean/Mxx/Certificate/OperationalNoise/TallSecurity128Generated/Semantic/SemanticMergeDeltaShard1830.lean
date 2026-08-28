import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge295620
def owner : Owner := ⟨.program ⟨257⟩, ⟨46870⟩⟩
def mergeEvent : Nat := 295620
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46409⟩⟩] } }
def rhsRaw : List Term := Proof.Events1154.exact295545RawTerms
def group : MergeGroup := .relation 295619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295619) (rhsResult := 295545)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46869⟩⟩) ⟨46409⟩ 295545) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46409⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], [⟨.program ⟨257⟩, ⟨46409⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295620

namespace LeftMerge295621
def owner : Owner := ⟨.program ⟨257⟩, ⟨46870⟩⟩
def mergeEvent : Nat := 295621
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩] } }
def leftRaw : List Term := Proof.Events1154.exact295612RawTerms
def rightRaw : List Term := Proof.Events1154.exact295548RawTerms
def group : MergeGroup := .operator 295612 295548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295612) (leftOrdinal := 0)
    (rightResult := 295548) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46869⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295621

namespace LeftMerge295635
def owner : Owner := ⟨.program ⟨257⟩, ⟨45812⟩⟩
def mergeEvent : Nat := 295635
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45809⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1154.exact295629RawTerms
def group : MergeGroup := .operator 295195 295629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 295629) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45809⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45809⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295635

namespace LeftMerge295690
def owner : Owner := ⟨.program ⟨257⟩, ⟨44915⟩⟩
def mergeEvent : Nat := 295690
def frameStart : Nat := 295672
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1155.exact295686RawTerms
def rightRaw : List Term := Proof.Events1155.exact295683RawTerms
def group : MergeGroup := .operator 295686 295683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295686) (leftOrdinal := 0)
    (rightResult := 295683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14631⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295690

namespace LeftMerge295720
def owner : Owner := ⟨.program ⟨257⟩, ⟨46708⟩⟩
def mergeEvent : Nat := 295720
def frameStart : Nat := 295672
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1155.exact295716RawTerms
def rightRaw : List Term := Proof.Events1155.exact295714RawTerms
def group : MergeGroup := .operator 295716 295714
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295716) (leftOrdinal := 0)
    (rightResult := 295714) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295720

namespace LeftMerge295743
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 295743
def frameStart : Nat := 295672
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events1155.exact295739RawTerms
def rightRaw : List Term := Proof.Events1155.exact295736RawTerms
def group : MergeGroup := .operator 295739 295736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295739) (leftOrdinal := 0)
    (rightResult := 295736) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295743

namespace LeftMerge295752
def owner : Owner := ⟨.program ⟨257⟩, ⟨46872⟩⟩
def mergeEvent : Nat := 295752
def frameStart : Nat := 295672
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩] } }
def leftRaw : List Term := Proof.Events1155.exact295748RawTerms
def rightRaw : List Term := Proof.Events1155.exact295705RawTerms
def group : MergeGroup := .operator 295748 295705
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295748) (leftOrdinal := 0)
    (rightResult := 295705) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46869⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295752

namespace LeftMerge295753
def owner : Owner := ⟨.program ⟨257⟩, ⟨46872⟩⟩
def mergeEvent : Nat := 295753
def frameStart : Nat := 295672
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩] } }
def leftRaw : List Term := Proof.Events1155.exact295748RawTerms
def rightRaw : List Term := Proof.Events1155.exact295705RawTerms
def group : MergeGroup := .operator 295748 295705
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295748) (leftOrdinal := 1)
    (rightResult := 295705) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46869⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295753

namespace LeftMerge295755
def owner : Owner := ⟨.program ⟨257⟩, ⟨46872⟩⟩
def mergeEvent : Nat := 295755
def frameStart : Nat := 295672
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46409⟩⟩] } }
def rhsRaw : List Term := Proof.Events1155.exact295702RawTerms
def group : MergeGroup := .relation 295754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295754) (rhsResult := 295702)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46869⟩⟩) ⟨46409⟩ 295702) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46409⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], [⟨.program ⟨257⟩, ⟨46409⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295755

namespace LeftMerge295763
def owner : Owner := ⟨.program ⟨257⟩, ⟨45390⟩⟩
def mergeEvent : Nat := 295763
def frameStart : Nat := 295672
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1155.exact295716RawTerms
def rightRaw : List Term := Proof.Events1155.exact295759RawTerms
def group : MergeGroup := .operator 295716 295759
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295716) (leftOrdinal := 0)
    (rightResult := 295759) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295763

namespace LeftMerge295780
def owner : Owner := ⟨.program ⟨257⟩, ⟨45812⟩⟩
def mergeEvent : Nat := 295780
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events1155.exact295777RawTerms
def group : MergeGroup := .relation 295779
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295779) (rhsResult := 295777)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 295778 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45809⟩⟩]⟩) (none) 295777) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295780

namespace LeftMerge295781
def owner : Owner := ⟨.program ⟨257⟩, ⟨45812⟩⟩
def mergeEvent : Nat := 295781
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩] } }
def rhsRaw : List Term := Proof.Events1155.exact295777RawTerms
def group : MergeGroup := .relation 295779
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295779) (rhsResult := 295777)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 295778 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45809⟩⟩]⟩) (none) 295777) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295781

namespace LeftMerge295782
def owner : Owner := ⟨.program ⟨257⟩, ⟨45812⟩⟩
def mergeEvent : Nat := 295782
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46409⟩⟩] } }
def rhsRaw : List Term := Proof.Events1155.exact295777RawTerms
def group : MergeGroup := .relation 295779
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295779) (rhsResult := 295777)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 295778 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45809⟩⟩]⟩) (none) 295777) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46409⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], [⟨.program ⟨257⟩, ⟨46409⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295782

namespace LeftMerge295783
def owner : Owner := ⟨.program ⟨257⟩, ⟨45812⟩⟩
def mergeEvent : Nat := 295783
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1155.exact295777RawTerms
def group : MergeGroup := .relation 295779
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 295779) (rhsResult := 295777)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45809⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 295778 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45809⟩⟩]⟩) (none) 295777) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295783

namespace LeftMerge295788
def owner : Owner := ⟨.program ⟨257⟩, ⟨46871⟩⟩
def mergeEvent : Nat := 295788
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46409⟩⟩] } }
def leftRaw : List Term := Proof.Events1155.exact295784RawTerms
def rightRaw : List Term := Proof.Events1154.exact295622RawTerms
def group : MergeGroup := .operator 295784 295622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295784) (leftOrdinal := 2)
    (rightResult := 295622) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46409⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46409⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], [⟨.program ⟨257⟩, ⟨46409⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge295788

namespace LeftMerge295789
def owner : Owner := ⟨.program ⟨257⟩, ⟨46871⟩⟩
def mergeEvent : Nat := 295789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩] } }
def leftRaw : List Term := Proof.Events1155.exact295784RawTerms
def rightRaw : List Term := Proof.Events1154.exact295622RawTerms
def group : MergeGroup := .operator 295784 295622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295784) (leftOrdinal := 1)
    (rightResult := 295622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46869⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge295789

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
