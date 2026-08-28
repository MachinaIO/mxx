import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge48820
def owner : Owner := ⟨.program ⟨257⟩, ⟨39030⟩⟩
def mergeEvent : Nat := 48820
def frameStart : Nat := 48728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩] } }
def leftRaw : List Term := Proof.Events190.exact48816RawTerms
def rightRaw : List Term := Proof.Events190.exact48773RawTerms
def group : MergeGroup := .operator 48816 48773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48816) (leftOrdinal := 0)
    (rightResult := 48773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39027⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48820

namespace LeftMerge48821
def owner : Owner := ⟨.program ⟨257⟩, ⟨39030⟩⟩
def mergeEvent : Nat := 48821
def frameStart : Nat := 48728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩] } }
def leftRaw : List Term := Proof.Events190.exact48816RawTerms
def rightRaw : List Term := Proof.Events190.exact48773RawTerms
def group : MergeGroup := .operator 48816 48773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48816) (leftOrdinal := 1)
    (rightResult := 48773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39027⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48821

namespace LeftMerge48823
def owner : Owner := ⟨.program ⟨257⟩, ⟨39030⟩⟩
def mergeEvent : Nat := 48823
def frameStart : Nat := 48728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38477⟩⟩] } }
def rhsRaw : List Term := Proof.Events190.exact48770RawTerms
def group : MergeGroup := .relation 48822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48822) (rhsResult := 48770)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39027⟩⟩) ⟨38477⟩ 48770) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38477⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48823

namespace LeftMerge48831
def owner : Owner := ⟨.program ⟨257⟩, ⟨37494⟩⟩
def mergeEvent : Nat := 48831
def frameStart : Nat := 48728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events190.exact48784RawTerms
def rightRaw : List Term := Proof.Events190.exact48827RawTerms
def group : MergeGroup := .operator 48784 48827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48784) (leftOrdinal := 0)
    (rightResult := 48827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37492⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48831

namespace LeftMerge48848
def owner : Owner := ⟨.program ⟨257⟩, ⟨37952⟩⟩
def mergeEvent : Nat := 48848
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events190.exact48845RawTerms
def group : MergeGroup := .relation 48847
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48847) (rhsResult := 48845)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48846 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩) (none) 48845) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48848

namespace LeftMerge48849
def owner : Owner := ⟨.program ⟨257⟩, ⟨37952⟩⟩
def mergeEvent : Nat := 48849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩] } }
def rhsRaw : List Term := Proof.Events190.exact48845RawTerms
def group : MergeGroup := .relation 48847
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48847) (rhsResult := 48845)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48846 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩) (none) 48845) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48849

namespace LeftMerge48850
def owner : Owner := ⟨.program ⟨257⟩, ⟨37952⟩⟩
def mergeEvent : Nat := 48850
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38477⟩⟩] } }
def rhsRaw : List Term := Proof.Events190.exact48845RawTerms
def group : MergeGroup := .relation 48847
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48847) (rhsResult := 48845)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48846 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩) (none) 48845) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38477⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48850

namespace LeftMerge48851
def owner : Owner := ⟨.program ⟨257⟩, ⟨37952⟩⟩
def mergeEvent : Nat := 48851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events190.exact48845RawTerms
def group : MergeGroup := .relation 48847
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48847) (rhsResult := 48845)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48846 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩) (none) 48845) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48851

namespace LeftMerge48856
def owner : Owner := ⟨.program ⟨257⟩, ⟨39029⟩⟩
def mergeEvent : Nat := 48856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38477⟩⟩] } }
def leftRaw : List Term := Proof.Events190.exact48852RawTerms
def rightRaw : List Term := Proof.Events190.exact48666RawTerms
def group : MergeGroup := .operator 48852 48666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48852) (leftOrdinal := 2)
    (rightResult := 48666) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38477⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38477⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48856

namespace LeftMerge48857
def owner : Owner := ⟨.program ⟨257⟩, ⟨39029⟩⟩
def mergeEvent : Nat := 48857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩] } }
def leftRaw : List Term := Proof.Events190.exact48852RawTerms
def rightRaw : List Term := Proof.Events190.exact48666RawTerms
def group : MergeGroup := .operator 48852 48666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48852) (leftOrdinal := 1)
    (rightResult := 48666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48857

namespace LeftMerge48865
def owner : Owner := ⟨.program ⟨257⟩, ⟨39511⟩⟩
def mergeEvent : Nat := 48865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩] } }
def leftRaw : List Term := Proof.Events190.exact48859RawTerms
def rightRaw : List Term := Proof.Events189.exact48582RawTerms
def group : MergeGroup := .operator 48859 48582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48859) (leftOrdinal := 0)
    (rightResult := 48582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39509⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48865

namespace LeftMerge48866
def owner : Owner := ⟨.program ⟨257⟩, ⟨39511⟩⟩
def mergeEvent : Nat := 48866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩] } }
def leftRaw : List Term := Proof.Events190.exact48859RawTerms
def rightRaw : List Term := Proof.Events189.exact48582RawTerms
def group : MergeGroup := .operator 48859 48582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48859) (leftOrdinal := 1)
    (rightResult := 48582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39509⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48866

namespace LeftMerge48868
def owner : Owner := ⟨.program ⟨257⟩, ⟨39511⟩⟩
def mergeEvent : Nat := 48868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38653⟩⟩] } }
def rhsRaw : List Term := Proof.Events189.exact48579RawTerms
def group : MergeGroup := .relation 48867
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48867) (rhsResult := 48579)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39509⟩⟩) ⟨38653⟩ 48579) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38653⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38653⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48868

namespace LeftMerge48882
def owner : Owner := ⟨.program ⟨257⟩, ⟨38339⟩⟩
def mergeEvent : Nat := 48882
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38336⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events190.exact48876RawTerms
def group : MergeGroup := .operator 46745 48876
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 48876) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38336⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38336⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48882

namespace LeftMerge49003
def owner : Owner := ⟨.program ⟨257⟩, ⟨38820⟩⟩
def mergeEvent : Nat := 49003
def frameStart : Nat := 48937
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events191.exact48999RawTerms
def rightRaw : List Term := Proof.Events191.exact48997RawTerms
def group : MergeGroup := .operator 48999 48997
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48999) (leftOrdinal := 0)
    (rightResult := 48997) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37492⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49003

namespace LeftMerge49015
def owner : Owner := ⟨.program ⟨257⟩, ⟨39510⟩⟩
def mergeEvent : Nat := 49015
def frameStart : Nat := 48937
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩] } }
def leftRaw : List Term := Proof.Events191.exact49011RawTerms
def rightRaw : List Term := Proof.Events191.exact48988RawTerms
def group : MergeGroup := .operator 49011 48988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49011) (leftOrdinal := 0)
    (rightResult := 48988) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39509⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49015

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
