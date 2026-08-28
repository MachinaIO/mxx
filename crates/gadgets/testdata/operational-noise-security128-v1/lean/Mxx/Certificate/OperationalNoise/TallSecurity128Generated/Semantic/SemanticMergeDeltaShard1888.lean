import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge304716
def owner : Owner := ⟨.program ⟨257⟩, ⟨47095⟩⟩
def mergeEvent : Nat := 304716
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46530⟩⟩] } }
def rhsRaw : List Term := Proof.Events1190.exact304704RawTerms
def group : MergeGroup := .relation 304715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304715) (rhsResult := 304704)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47093⟩⟩) ⟨46530⟩ 304704) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46530⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304716

namespace LeftMerge304730
def owner : Owner := ⟨.program ⟨257⟩, ⟨46015⟩⟩
def mergeEvent : Nat := 304730
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1190.exact304724RawTerms
def group : MergeGroup := .operator 295195 304724
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 304724) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46012⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304730

namespace LeftMerge304827
def owner : Owner := ⟨.program ⟨257⟩, ⟨46788⟩⟩
def mergeEvent : Nat := 304827
def frameStart : Nat := 304773
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1190.exact304823RawTerms
def rightRaw : List Term := Proof.Events1190.exact304821RawTerms
def group : MergeGroup := .operator 304823 304821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304823) (leftOrdinal := 0)
    (rightResult := 304821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304827

namespace LeftMerge304839
def owner : Owner := ⟨.program ⟨257⟩, ⟨47094⟩⟩
def mergeEvent : Nat := 304839
def frameStart : Nat := 304773
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩] } }
def leftRaw : List Term := Proof.Events1190.exact304835RawTerms
def rightRaw : List Term := Proof.Events1190.exact304812RawTerms
def group : MergeGroup := .operator 304835 304812
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304835) (leftOrdinal := 0)
    (rightResult := 304812) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47093⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304839

namespace LeftMerge304840
def owner : Owner := ⟨.program ⟨257⟩, ⟨47094⟩⟩
def mergeEvent : Nat := 304840
def frameStart : Nat := 304773
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩] } }
def leftRaw : List Term := Proof.Events1190.exact304835RawTerms
def rightRaw : List Term := Proof.Events1190.exact304812RawTerms
def group : MergeGroup := .operator 304835 304812
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304835) (leftOrdinal := 1)
    (rightResult := 304812) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47093⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304840

namespace LeftMerge304842
def owner : Owner := ⟨.program ⟨257⟩, ⟨47094⟩⟩
def mergeEvent : Nat := 304842
def frameStart : Nat := 304773
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46530⟩⟩] } }
def rhsRaw : List Term := Proof.Events1190.exact304809RawTerms
def group : MergeGroup := .relation 304841
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304841) (rhsResult := 304809)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47093⟩⟩) ⟨46530⟩ 304809) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46530⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304842

namespace LeftMerge304850
def owner : Owner := ⟨.program ⟨257⟩, ⟨45551⟩⟩
def mergeEvent : Nat := 304850
def frameStart : Nat := 304773
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1190.exact304823RawTerms
def rightRaw : List Term := Proof.Events1190.exact304846RawTerms
def group : MergeGroup := .operator 304823 304846
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304823) (leftOrdinal := 0)
    (rightResult := 304846) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304850

namespace LeftMerge304867
def owner : Owner := ⟨.program ⟨257⟩, ⟨46015⟩⟩
def mergeEvent : Nat := 304867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }
def rhsRaw : List Term := Proof.Events1190.exact304864RawTerms
def group : MergeGroup := .relation 304866
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304866) (rhsResult := 304864)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 304865 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩) (none) 304864) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304867

namespace LeftMerge304868
def owner : Owner := ⟨.program ⟨257⟩, ⟨46015⟩⟩
def mergeEvent : Nat := 304868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩] } }
def rhsRaw : List Term := Proof.Events1190.exact304864RawTerms
def group : MergeGroup := .relation 304866
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304866) (rhsResult := 304864)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 304865 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩) (none) 304864) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304868

namespace LeftMerge304869
def owner : Owner := ⟨.program ⟨257⟩, ⟨46015⟩⟩
def mergeEvent : Nat := 304869
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46530⟩⟩] } }
def rhsRaw : List Term := Proof.Events1190.exact304864RawTerms
def group : MergeGroup := .relation 304866
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304866) (rhsResult := 304864)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 304865 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩) (none) 304864) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46530⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304869

namespace LeftMerge304870
def owner : Owner := ⟨.program ⟨257⟩, ⟨46015⟩⟩
def mergeEvent : Nat := 304870
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1190.exact304864RawTerms
def group : MergeGroup := .relation 304866
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304866) (rhsResult := 304864)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 304865 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩) (none) 304864) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304870

namespace LeftMerge304875
def owner : Owner := ⟨.program ⟨257⟩, ⟨47096⟩⟩
def mergeEvent : Nat := 304875
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩] } }
def leftRaw : List Term := Proof.Events1190.exact304871RawTerms
def rightRaw : List Term := Proof.Events1190.exact304717RawTerms
def group : MergeGroup := .operator 304871 304717
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304871) (leftOrdinal := 0)
    (rightResult := 304717) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304875

namespace LeftMerge304876
def owner : Owner := ⟨.program ⟨257⟩, ⟨47096⟩⟩
def mergeEvent : Nat := 304876
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46530⟩⟩] } }
def leftRaw : List Term := Proof.Events1190.exact304871RawTerms
def rightRaw : List Term := Proof.Events1190.exact304717RawTerms
def group : MergeGroup := .operator 304871 304717
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304871) (leftOrdinal := 2)
    (rightResult := 304717) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46530⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46530⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304876

namespace LeftMerge304884
def owner : Owner := ⟨.program ⟨257⟩, ⟨47097⟩⟩
def mergeEvent : Nat := 304884
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }
def leftRaw : List Term := Proof.Events1190.exact304878RawTerms
def rightRaw : List Term := Proof.Events060.exact15562RawTerms
def group : MergeGroup := .operator 304878 15562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304878) (leftOrdinal := 0)
    (rightResult := 15562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304884

namespace LeftMerge304885
def owner : Owner := ⟨.program ⟨257⟩, ⟨47097⟩⟩
def mergeEvent : Nat := 304885
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }
def leftRaw : List Term := Proof.Events1190.exact304878RawTerms
def rightRaw : List Term := Proof.Events060.exact15562RawTerms
def group : MergeGroup := .operator 304878 15562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304878) (leftOrdinal := 1)
    (rightResult := 15562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304885

namespace LeftMerge304887
def owner : Owner := ⟨.program ⟨257⟩, ⟨47097⟩⟩
def mergeEvent : Nat := 304887
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15555RawTerms
def group : MergeGroup := .relation 304886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304886) (rhsResult := 15555)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7151⟩⟩) ⟨7041⟩ 15555) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304887

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
