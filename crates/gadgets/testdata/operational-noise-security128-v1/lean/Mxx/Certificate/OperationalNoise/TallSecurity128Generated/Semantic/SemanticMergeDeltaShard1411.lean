import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge229652
def owner : Owner := ⟨.program ⟨257⟩, ⟨22362⟩⟩
def mergeEvent : Nat := 229652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21086⟩⟩, ⟨.program ⟨257⟩, ⟨21470⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22923⟩⟩] } }
def rhsRaw : List Term := Proof.Events897.exact229647RawTerms
def group : MergeGroup := .relation 229649
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229649) (rhsResult := 229647)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 229648 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22359⟩⟩]⟩) (none) 229647) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21086⟩⟩, ⟨.program ⟨257⟩, ⟨21470⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22923⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21086⟩⟩, ⟨.program ⟨257⟩, ⟨21470⟩⟩], [⟨.program ⟨257⟩, ⟨22923⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229652

namespace LeftMerge229653
def owner : Owner := ⟨.program ⟨257⟩, ⟨22362⟩⟩
def mergeEvent : Nat := 229653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events897.exact229647RawTerms
def group : MergeGroup := .relation 229649
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229649) (rhsResult := 229647)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 229648 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22359⟩⟩]⟩) (none) 229647) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229653

namespace LeftMerge229658
def owner : Owner := ⟨.program ⟨257⟩, ⟨23430⟩⟩
def mergeEvent : Nat := 229658
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21086⟩⟩, ⟨.program ⟨257⟩, ⟨21470⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22923⟩⟩] } }
def leftRaw : List Term := Proof.Events897.exact229654RawTerms
def rightRaw : List Term := Proof.Events896.exact229468RawTerms
def group : MergeGroup := .operator 229654 229468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229654) (leftOrdinal := 2)
    (rightResult := 229468) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21086⟩⟩, ⟨.program ⟨257⟩, ⟨21470⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22923⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21086⟩⟩, ⟨.program ⟨257⟩, ⟨21470⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22923⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21086⟩⟩, ⟨.program ⟨257⟩, ⟨21470⟩⟩], [⟨.program ⟨257⟩, ⟨22923⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229658

namespace LeftMerge229659
def owner : Owner := ⟨.program ⟨257⟩, ⟨23430⟩⟩
def mergeEvent : Nat := 229659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23428⟩⟩] } }
def leftRaw : List Term := Proof.Events897.exact229654RawTerms
def rightRaw : List Term := Proof.Events896.exact229468RawTerms
def group : MergeGroup := .operator 229654 229468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229654) (leftOrdinal := 1)
    (rightResult := 229468) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23428⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23428⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23428⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229659

namespace LeftMerge229667
def owner : Owner := ⟨.program ⟨257⟩, ⟨23843⟩⟩
def mergeEvent : Nat := 229667
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩] } }
def leftRaw : List Term := Proof.Events897.exact229661RawTerms
def rightRaw : List Term := Proof.Events896.exact229384RawTerms
def group : MergeGroup := .operator 229661 229384
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229661) (leftOrdinal := 0)
    (rightResult := 229384) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23841⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229667

namespace LeftMerge229668
def owner : Owner := ⟨.program ⟨257⟩, ⟨23843⟩⟩
def mergeEvent : Nat := 229668
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩] } }
def leftRaw : List Term := Proof.Events897.exact229661RawTerms
def rightRaw : List Term := Proof.Events896.exact229384RawTerms
def group : MergeGroup := .operator 229661 229384
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229661) (leftOrdinal := 1)
    (rightResult := 229384) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23841⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229668

namespace LeftMerge229670
def owner : Owner := ⟨.program ⟨257⟩, ⟨23843⟩⟩
def mergeEvent : Nat := 229670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23072⟩⟩] } }
def rhsRaw : List Term := Proof.Events896.exact229381RawTerms
def group : MergeGroup := .relation 229669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229669) (rhsResult := 229381)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23841⟩⟩) ⟨23072⟩ 229381) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23072⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23072⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229670

namespace LeftMerge229684
def owner : Owner := ⟨.program ⟨257⟩, ⟨22659⟩⟩
def mergeEvent : Nat := 229684
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22656⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events897.exact229678RawTerms
def group : MergeGroup := .operator 222245 229678
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 229678) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22656⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22656⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229684

namespace LeftMerge229805
def owner : Owner := ⟨.program ⟨257⟩, ⟨23284⟩⟩
def mergeEvent : Nat := 229805
def frameStart : Nat := 229739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events897.exact229801RawTerms
def rightRaw : List Term := Proof.Events897.exact229799RawTerms
def group : MergeGroup := .operator 229801 229799
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229801) (leftOrdinal := 0)
    (rightResult := 229799) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229805

namespace LeftMerge229817
def owner : Owner := ⟨.program ⟨257⟩, ⟨23842⟩⟩
def mergeEvent : Nat := 229817
def frameStart : Nat := 229739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩] } }
def leftRaw : List Term := Proof.Events897.exact229813RawTerms
def rightRaw : List Term := Proof.Events897.exact229790RawTerms
def group : MergeGroup := .operator 229813 229790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229813) (leftOrdinal := 0)
    (rightResult := 229790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23841⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229817

namespace LeftMerge229818
def owner : Owner := ⟨.program ⟨257⟩, ⟨23842⟩⟩
def mergeEvent : Nat := 229818
def frameStart : Nat := 229739
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩] } }
def leftRaw : List Term := Proof.Events897.exact229813RawTerms
def rightRaw : List Term := Proof.Events897.exact229790RawTerms
def group : MergeGroup := .operator 229813 229790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229813) (leftOrdinal := 1)
    (rightResult := 229790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23841⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229818

namespace LeftMerge229820
def owner : Owner := ⟨.program ⟨257⟩, ⟨23842⟩⟩
def mergeEvent : Nat := 229820
def frameStart : Nat := 229739
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23072⟩⟩] } }
def rhsRaw : List Term := Proof.Events897.exact229787RawTerms
def group : MergeGroup := .relation 229819
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229819) (rhsResult := 229787)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23841⟩⟩) ⟨23072⟩ 229787) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23072⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23072⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229820

namespace LeftMerge229828
def owner : Owner := ⟨.program ⟨257⟩, ⟨22069⟩⟩
def mergeEvent : Nat := 229828
def frameStart : Nat := 229739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events897.exact229801RawTerms
def rightRaw : List Term := Proof.Events897.exact229824RawTerms
def group : MergeGroup := .operator 229801 229824
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 229801) (leftOrdinal := 0)
    (rightResult := 229824) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229828

namespace LeftMerge229845
def owner : Owner := ⟨.program ⟨257⟩, ⟨22659⟩⟩
def mergeEvent : Nat := 229845
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }
def rhsRaw : List Term := Proof.Events897.exact229842RawTerms
def group : MergeGroup := .relation 229844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229844) (rhsResult := 229842)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 229843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22656⟩⟩]⟩) (none) 229842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229845

namespace LeftMerge229846
def owner : Owner := ⟨.program ⟨257⟩, ⟨22659⟩⟩
def mergeEvent : Nat := 229846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩] } }
def rhsRaw : List Term := Proof.Events897.exact229842RawTerms
def group : MergeGroup := .relation 229844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229844) (rhsResult := 229842)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 229843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22656⟩⟩]⟩) (none) 229842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23841⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge229846

namespace LeftMerge229847
def owner : Owner := ⟨.program ⟨257⟩, ⟨22659⟩⟩
def mergeEvent : Nat := 229847
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23072⟩⟩] } }
def rhsRaw : List Term := Proof.Events897.exact229842RawTerms
def group : MergeGroup := .relation 229844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 229844) (rhsResult := 229842)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 229843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22656⟩⟩]⟩) (none) 229842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23072⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23072⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229847

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
