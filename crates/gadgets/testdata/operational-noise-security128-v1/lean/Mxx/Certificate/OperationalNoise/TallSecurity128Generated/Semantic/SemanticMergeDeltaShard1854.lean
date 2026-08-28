import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge299658
def owner : Owner := ⟨.program ⟨257⟩, ⟨61352⟩⟩
def mergeEvent : Nat := 299658
def frameStart : Nat := 299578
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩] } }
def leftRaw : List Term := Proof.Events1170.exact299654RawTerms
def rightRaw : List Term := Proof.Events1170.exact299611RawTerms
def group : MergeGroup := .operator 299654 299611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299654) (leftOrdinal := 0)
    (rightResult := 299611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61349⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299658

namespace LeftMerge299659
def owner : Owner := ⟨.program ⟨257⟩, ⟨61352⟩⟩
def mergeEvent : Nat := 299659
def frameStart : Nat := 299578
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩] } }
def leftRaw : List Term := Proof.Events1170.exact299654RawTerms
def rightRaw : List Term := Proof.Events1170.exact299611RawTerms
def group : MergeGroup := .operator 299654 299611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299654) (leftOrdinal := 1)
    (rightResult := 299611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299659

namespace LeftMerge299661
def owner : Owner := ⟨.program ⟨257⟩, ⟨61352⟩⟩
def mergeEvent : Nat := 299661
def frameStart : Nat := 299578
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60889⟩⟩] } }
def rhsRaw : List Term := Proof.Events1170.exact299608RawTerms
def group : MergeGroup := .relation 299660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299660) (rhsResult := 299608)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61349⟩⟩) ⟨60889⟩ 299608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60889⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], [⟨.program ⟨257⟩, ⟨60889⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299661

namespace LeftMerge299669
def owner : Owner := ⟨.program ⟨257⟩, ⟨59750⟩⟩
def mergeEvent : Nat := 299669
def frameStart : Nat := 299578
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1170.exact299622RawTerms
def rightRaw : List Term := Proof.Events1170.exact299665RawTerms
def group : MergeGroup := .operator 299622 299665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299622) (leftOrdinal := 0)
    (rightResult := 299665) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299669

namespace LeftMerge299686
def owner : Owner := ⟨.program ⟨257⟩, ⟨60292⟩⟩
def mergeEvent : Nat := 299686
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events1170.exact299683RawTerms
def group : MergeGroup := .relation 299685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299685) (rhsResult := 299683)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 299684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60289⟩⟩]⟩) (none) 299683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299686

namespace LeftMerge299687
def owner : Owner := ⟨.program ⟨257⟩, ⟨60292⟩⟩
def mergeEvent : Nat := 299687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩] } }
def rhsRaw : List Term := Proof.Events1170.exact299683RawTerms
def group : MergeGroup := .relation 299685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299685) (rhsResult := 299683)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 299684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60289⟩⟩]⟩) (none) 299683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299687

namespace LeftMerge299688
def owner : Owner := ⟨.program ⟨257⟩, ⟨60292⟩⟩
def mergeEvent : Nat := 299688
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60889⟩⟩] } }
def rhsRaw : List Term := Proof.Events1170.exact299683RawTerms
def group : MergeGroup := .relation 299685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299685) (rhsResult := 299683)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 299684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60289⟩⟩]⟩) (none) 299683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60889⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], [⟨.program ⟨257⟩, ⟨60889⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299688

namespace LeftMerge299689
def owner : Owner := ⟨.program ⟨257⟩, ⟨60292⟩⟩
def mergeEvent : Nat := 299689
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1170.exact299683RawTerms
def group : MergeGroup := .relation 299685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299685) (rhsResult := 299683)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 299684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60289⟩⟩]⟩) (none) 299683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299689

namespace LeftMerge299694
def owner : Owner := ⟨.program ⟨257⟩, ⟨61351⟩⟩
def mergeEvent : Nat := 299694
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60889⟩⟩] } }
def leftRaw : List Term := Proof.Events1170.exact299690RawTerms
def rightRaw : List Term := Proof.Events1170.exact299528RawTerms
def group : MergeGroup := .operator 299690 299528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299690) (leftOrdinal := 2)
    (rightResult := 299528) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60889⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60889⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25130⟩⟩, ⟨.program ⟨257⟩, ⟨59215⟩⟩], [⟨.program ⟨257⟩, ⟨60889⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299694

namespace LeftMerge299695
def owner : Owner := ⟨.program ⟨257⟩, ⟨61351⟩⟩
def mergeEvent : Nat := 299695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩] } }
def leftRaw : List Term := Proof.Events1170.exact299690RawTerms
def rightRaw : List Term := Proof.Events1170.exact299528RawTerms
def group : MergeGroup := .operator 299690 299528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299690) (leftOrdinal := 1)
    (rightResult := 299528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299695

namespace LeftMerge299703
def owner : Owner := ⟨.program ⟨257⟩, ⟨61584⟩⟩
def mergeEvent : Nat := 299703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61582⟩⟩] } }
def leftRaw : List Term := Proof.Events1170.exact299697RawTerms
def rightRaw : List Term := Proof.Events1169.exact299444RawTerms
def group : MergeGroup := .operator 299697 299444
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299697) (leftOrdinal := 0)
    (rightResult := 299444) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61582⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61582⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299703

namespace LeftMerge299704
def owner : Owner := ⟨.program ⟨257⟩, ⟨61584⟩⟩
def mergeEvent : Nat := 299704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61582⟩⟩] } }
def leftRaw : List Term := Proof.Events1170.exact299697RawTerms
def rightRaw : List Term := Proof.Events1169.exact299444RawTerms
def group : MergeGroup := .operator 299697 299444
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299697) (leftOrdinal := 1)
    (rightResult := 299444) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61582⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61582⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299704

namespace LeftMerge299706
def owner : Owner := ⟨.program ⟨257⟩, ⟨61584⟩⟩
def mergeEvent : Nat := 299706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61011⟩⟩] } }
def rhsRaw : List Term := Proof.Events1169.exact299441RawTerms
def group : MergeGroup := .relation 299705
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299705) (rhsResult := 299441)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61582⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61582⟩⟩) ⟨61011⟩ 299441) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61011⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨61011⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299706

namespace LeftMerge299720
def owner : Owner := ⟨.program ⟨257⟩, ⟨60499⟩⟩
def mergeEvent : Nat := 299720
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60496⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1170.exact299714RawTerms
def group : MergeGroup := .operator 295195 299714
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 299714) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60496⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60496⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299720

namespace LeftMerge299817
def owner : Owner := ⟨.program ⟨257⟩, ⟨61268⟩⟩
def mergeEvent : Nat := 299817
def frameStart : Nat := 299763
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1171.exact299813RawTerms
def rightRaw : List Term := Proof.Events1171.exact299811RawTerms
def group : MergeGroup := .operator 299813 299811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299813) (leftOrdinal := 0)
    (rightResult := 299811) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299817

namespace LeftMerge299829
def owner : Owner := ⟨.program ⟨257⟩, ⟨61583⟩⟩
def mergeEvent : Nat := 299829
def frameStart : Nat := 299763
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61582⟩⟩] } }
def leftRaw : List Term := Proof.Events1171.exact299825RawTerms
def rightRaw : List Term := Proof.Events1171.exact299802RawTerms
def group : MergeGroup := .operator 299825 299802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299825) (leftOrdinal := 0)
    (rightResult := 299802) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61582⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61582⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299829

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
