import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge125792
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 125792
def frameStart : Nat := 125709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events491.exact125788RawTerms
def rightRaw : List Term := Proof.Events491.exact125785RawTerms
def group : MergeGroup := .operator 125788 125785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125788) (leftOrdinal := 0)
    (rightResult := 125785) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125792

namespace LeftMerge125801
def owner : Owner := ⟨.program ⟨257⟩, ⟨55458⟩⟩
def mergeEvent : Nat := 125801
def frameStart : Nat := 125709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩] } }
def leftRaw : List Term := Proof.Events491.exact125797RawTerms
def rightRaw : List Term := Proof.Events491.exact125754RawTerms
def group : MergeGroup := .operator 125797 125754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125797) (leftOrdinal := 0)
    (rightResult := 125754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55455⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125801

namespace LeftMerge125802
def owner : Owner := ⟨.program ⟨257⟩, ⟨55458⟩⟩
def mergeEvent : Nat := 125802
def frameStart : Nat := 125709
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩] } }
def leftRaw : List Term := Proof.Events491.exact125797RawTerms
def rightRaw : List Term := Proof.Events491.exact125754RawTerms
def group : MergeGroup := .operator 125797 125754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125797) (leftOrdinal := 1)
    (rightResult := 125754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55455⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125802

namespace LeftMerge125804
def owner : Owner := ⟨.program ⟨257⟩, ⟨55458⟩⟩
def mergeEvent : Nat := 125804
def frameStart : Nat := 125709
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54965⟩⟩] } }
def rhsRaw : List Term := Proof.Events491.exact125751RawTerms
def group : MergeGroup := .relation 125803
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125803) (rhsResult := 125751)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55455⟩⟩) ⟨54965⟩ 125751) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54965⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], [⟨.program ⟨257⟩, ⟨54965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125804

namespace LeftMerge125812
def owner : Owner := ⟨.program ⟨257⟩, ⟨53838⟩⟩
def mergeEvent : Nat := 125812
def frameStart : Nat := 125709
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events491.exact125765RawTerms
def rightRaw : List Term := Proof.Events491.exact125808RawTerms
def group : MergeGroup := .operator 125765 125808
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125765) (leftOrdinal := 0)
    (rightResult := 125808) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53836⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125812

namespace LeftMerge125829
def owner : Owner := ⟨.program ⟨257⟩, ⟨54392⟩⟩
def mergeEvent : Nat := 125829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events491.exact125826RawTerms
def group : MergeGroup := .relation 125828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125828) (rhsResult := 125826)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 125827 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54389⟩⟩]⟩) (none) 125826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125829

namespace LeftMerge125830
def owner : Owner := ⟨.program ⟨257⟩, ⟨54392⟩⟩
def mergeEvent : Nat := 125830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩] } }
def rhsRaw : List Term := Proof.Events491.exact125826RawTerms
def group : MergeGroup := .relation 125828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125828) (rhsResult := 125826)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 125827 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54389⟩⟩]⟩) (none) 125826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125830

namespace LeftMerge125831
def owner : Owner := ⟨.program ⟨257⟩, ⟨54392⟩⟩
def mergeEvent : Nat := 125831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54965⟩⟩] } }
def rhsRaw : List Term := Proof.Events491.exact125826RawTerms
def group : MergeGroup := .relation 125828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125828) (rhsResult := 125826)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 125827 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54389⟩⟩]⟩) (none) 125826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54965⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], [⟨.program ⟨257⟩, ⟨54965⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125831

namespace LeftMerge125832
def owner : Owner := ⟨.program ⟨257⟩, ⟨54392⟩⟩
def mergeEvent : Nat := 125832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨53836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events491.exact125826RawTerms
def group : MergeGroup := .relation 125828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125828) (rhsResult := 125826)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 125827 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54389⟩⟩]⟩) (none) 125826) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨53836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125832

namespace LeftMerge125837
def owner : Owner := ⟨.program ⟨257⟩, ⟨55457⟩⟩
def mergeEvent : Nat := 125837
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54965⟩⟩] } }
def leftRaw : List Term := Proof.Events491.exact125833RawTerms
def rightRaw : List Term := Proof.Events490.exact125647RawTerms
def group : MergeGroup := .operator 125833 125647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125833) (leftOrdinal := 2)
    (rightResult := 125647) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54965⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54965⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24722⟩⟩, ⟨.program ⟨257⟩, ⟨53417⟩⟩], [⟨.program ⟨257⟩, ⟨54965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125837

namespace LeftMerge125838
def owner : Owner := ⟨.program ⟨257⟩, ⟨55457⟩⟩
def mergeEvent : Nat := 125838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩] } }
def leftRaw : List Term := Proof.Events491.exact125833RawTerms
def rightRaw : List Term := Proof.Events490.exact125647RawTerms
def group : MergeGroup := .operator 125833 125647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125833) (leftOrdinal := 1)
    (rightResult := 125647) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55455⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125838

namespace LeftMerge125846
def owner : Owner := ⟨.program ⟨257⟩, ⟨55810⟩⟩
def mergeEvent : Nat := 125846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55808⟩⟩] } }
def leftRaw : List Term := Proof.Events491.exact125840RawTerms
def rightRaw : List Term := Proof.Events490.exact125563RawTerms
def group : MergeGroup := .operator 125840 125563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125840) (leftOrdinal := 0)
    (rightResult := 125563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55808⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55808⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125846

namespace LeftMerge125847
def owner : Owner := ⟨.program ⟨257⟩, ⟨55810⟩⟩
def mergeEvent : Nat := 125847
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨53836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55808⟩⟩] } }
def leftRaw : List Term := Proof.Events491.exact125840RawTerms
def rightRaw : List Term := Proof.Events490.exact125563RawTerms
def group : MergeGroup := .operator 125840 125563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125840) (leftOrdinal := 1)
    (rightResult := 125563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨53836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55808⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨53836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55808⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125847

namespace LeftMerge125849
def owner : Owner := ⟨.program ⟨257⟩, ⟨55810⟩⟩
def mergeEvent : Nat := 125849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨53836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55105⟩⟩] } }
def rhsRaw : List Term := Proof.Events490.exact125560RawTerms
def group : MergeGroup := .relation 125848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 125848) (rhsResult := 125560)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨53836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55808⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55808⟩⟩) ⟨55105⟩ 125560) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55105⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨53836⟩⟩], [⟨.program ⟨257⟩, ⟨55105⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge125849

namespace LeftMerge125863
def owner : Owner := ⟨.program ⟨257⟩, ⟨54659⟩⟩
def mergeEvent : Nat := 125863
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54656⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events491.exact125857RawTerms
def group : MergeGroup := .operator 119870 125857
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 125857) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54656⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54656⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125863

namespace LeftMerge125984
def owner : Owner := ⟨.program ⟨257⟩, ⟨55332⟩⟩
def mergeEvent : Nat := 125984
def frameStart : Nat := 125918
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events492.exact125980RawTerms
def rightRaw : List Term := Proof.Events492.exact125978RawTerms
def group : MergeGroup := .operator 125980 125978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 125980) (leftOrdinal := 0)
    (rightResult := 125978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53836⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge125984

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
