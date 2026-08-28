import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge18847
def owner : Owner := ⟨.program ⟨257⟩, ⟨40465⟩⟩
def mergeEvent : Nat := 18847
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41057⟩⟩] } }
def rhsRaw : List Term := Proof.Events073.exact18844RawTerms
def group : MergeGroup := .relation 18846
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18846) (rhsResult := 18844)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40462⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 18845 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40462⟩⟩]⟩) (none) 18844) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41057⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], [⟨.program ⟨257⟩, ⟨41057⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18847

namespace LeftMerge18848
def owner : Owner := ⟨.program ⟨257⟩, ⟨40465⟩⟩
def mergeEvent : Nat := 18848
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩] } }
def rhsRaw : List Term := Proof.Events073.exact18844RawTerms
def group : MergeGroup := .relation 18846
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18846) (rhsResult := 18844)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40462⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 18845 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40462⟩⟩]⟩) (none) 18844) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18848

namespace LeftMerge18849
def owner : Owner := ⟨.program ⟨257⟩, ⟨40465⟩⟩
def mergeEvent : Nat := 18849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events073.exact18844RawTerms
def group : MergeGroup := .relation 18846
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18846) (rhsResult := 18844)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40462⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 18845 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40462⟩⟩]⟩) (none) 18844) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18849

namespace LeftMerge18850
def owner : Owner := ⟨.program ⟨257⟩, ⟨40465⟩⟩
def mergeEvent : Nat := 18850
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events073.exact18844RawTerms
def group : MergeGroup := .relation 18846
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18846) (rhsResult := 18844)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40462⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 18845 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40462⟩⟩]⟩) (none) 18844) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18850

namespace LeftMerge18855
def owner : Owner := ⟨.program ⟨257⟩, ⟨41525⟩⟩
def mergeEvent : Nat := 18855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41057⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18851RawTerms
def rightRaw : List Term := Proof.Events072.exact18665RawTerms
def group : MergeGroup := .operator 18851 18665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18851) (leftOrdinal := 2)
    (rightResult := 18665) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41057⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41057⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], [⟨.program ⟨257⟩, ⟨41057⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18855

namespace LeftMerge18856
def owner : Owner := ⟨.program ⟨257⟩, ⟨41525⟩⟩
def mergeEvent : Nat := 18856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18851RawTerms
def rightRaw : List Term := Proof.Events072.exact18665RawTerms
def group : MergeGroup := .operator 18851 18665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18851) (leftOrdinal := 1)
    (rightResult := 18665) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18856

namespace LeftMerge18864
def owner : Owner := ⟨.program ⟨257⟩, ⟨41773⟩⟩
def mergeEvent : Nat := 18864
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41771⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18858RawTerms
def rightRaw : List Term := Proof.Events072.exact18562RawTerms
def group : MergeGroup := .operator 18858 18562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18858) (leftOrdinal := 1)
    (rightResult := 18562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41771⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41771⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18864

namespace LeftMerge18866
def owner : Owner := ⟨.program ⟨257⟩, ⟨41773⟩⟩
def mergeEvent : Nat := 18866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41183⟩⟩] } }
def rhsRaw : List Term := Proof.Events072.exact18559RawTerms
def group : MergeGroup := .relation 18865
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18865) (rhsResult := 18559)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41771⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41771⟩⟩) ⟨41183⟩ 18559) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41183⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18866

namespace LeftMerge18867
def owner : Owner := ⟨.program ⟨257⟩, ⟨41773⟩⟩
def mergeEvent : Nat := 18867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41771⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18858RawTerms
def rightRaw : List Term := Proof.Events072.exact18562RawTerms
def group : MergeGroup := .operator 18858 18562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18858) (leftOrdinal := 0)
    (rightResult := 18562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41771⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41771⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18867

namespace LeftMerge18881
def owner : Owner := ⟨.program ⟨257⟩, ⟨40685⟩⟩
def mergeEvent : Nat := 18881
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40682⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events073.exact18875RawTerms
def group : MergeGroup := .operator 17169 18875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 18875) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40682⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40682⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18881

namespace LeftMerge19002
def owner : Owner := ⟨.program ⟨257⟩, ⟨41432⟩⟩
def mergeEvent : Nat := 19002
def frameStart : Nat := 18936
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events074.exact18998RawTerms
def rightRaw : List Term := Proof.Events074.exact18996RawTerms
def group : MergeGroup := .operator 18998 18996
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18998) (leftOrdinal := 0)
    (rightResult := 18996) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19002

namespace LeftMerge19014
def owner : Owner := ⟨.program ⟨257⟩, ⟨41772⟩⟩
def mergeEvent : Nat := 19014
def frameStart : Nat := 18936
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41771⟩⟩] } }
def leftRaw : List Term := Proof.Events074.exact19010RawTerms
def rightRaw : List Term := Proof.Events074.exact18987RawTerms
def group : MergeGroup := .operator 19010 18987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19010) (leftOrdinal := 1)
    (rightResult := 18987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41771⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41771⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19014

namespace LeftMerge19016
def owner : Owner := ⟨.program ⟨257⟩, ⟨41772⟩⟩
def mergeEvent : Nat := 19016
def frameStart : Nat := 18936
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41183⟩⟩] } }
def rhsRaw : List Term := Proof.Events074.exact18984RawTerms
def group : MergeGroup := .relation 19015
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19015) (rhsResult := 18984)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41771⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41771⟩⟩) ⟨41183⟩ 18984) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41183⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19016

namespace LeftMerge19017
def owner : Owner := ⟨.program ⟨257⟩, ⟨41772⟩⟩
def mergeEvent : Nat := 19017
def frameStart : Nat := 18936
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41771⟩⟩] } }
def leftRaw : List Term := Proof.Events074.exact19010RawTerms
def rightRaw : List Term := Proof.Events074.exact18987RawTerms
def group : MergeGroup := .operator 19010 18987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19010) (leftOrdinal := 0)
    (rightResult := 18987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41771⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41771⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19017

namespace LeftMerge19025
def owner : Owner := ⟨.program ⟨257⟩, ⟨40206⟩⟩
def mergeEvent : Nat := 19025
def frameStart : Nat := 18936
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40205⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events074.exact18998RawTerms
def rightRaw : List Term := Proof.Events074.exact19021RawTerms
def group : MergeGroup := .operator 18998 19021
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18998) (leftOrdinal := 0)
    (rightResult := 19021) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40205⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19025

namespace LeftMerge19042
def owner : Owner := ⟨.program ⟨257⟩, ⟨40685⟩⟩
def mergeEvent : Nat := 19042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41183⟩⟩] } }
def rhsRaw : List Term := Proof.Events074.exact19039RawTerms
def group : MergeGroup := .relation 19041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19041) (rhsResult := 19039)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40682⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19040 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40682⟩⟩]⟩) (none) 19039) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19042

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
