import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge23829
def owner : Owner := ⟨.program ⟨257⟩, ⟨52426⟩⟩
def mergeEvent : Nat := 23829
def frameStart : Nat := 23737
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact23825RawTerms
def rightRaw : List Term := Proof.Events092.exact23782RawTerms
def group : MergeGroup := .operator 23825 23782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23825) (leftOrdinal := 1)
    (rightResult := 23782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52423⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23829

namespace LeftMerge23831
def owner : Owner := ⟨.program ⟨257⟩, ⟨52426⟩⟩
def mergeEvent : Nat := 23831
def frameStart : Nat := 23737
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51957⟩⟩] } }
def rhsRaw : List Term := Proof.Events092.exact23779RawTerms
def group : MergeGroup := .relation 23830
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23830) (rhsResult := 23779)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52423⟩⟩) ⟨51957⟩ 23779) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51957⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], [⟨.program ⟨257⟩, ⟨51957⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23831

namespace LeftMerge23832
def owner : Owner := ⟨.program ⟨257⟩, ⟨52426⟩⟩
def mergeEvent : Nat := 23832
def frameStart : Nat := 23737
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact23825RawTerms
def rightRaw : List Term := Proof.Events092.exact23782RawTerms
def group : MergeGroup := .operator 23825 23782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23825) (leftOrdinal := 0)
    (rightResult := 23782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52423⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23832

namespace LeftMerge23840
def owner : Owner := ⟨.program ⟨257⟩, ⟨50820⟩⟩
def mergeEvent : Nat := 23840
def frameStart : Nat := 23737
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events092.exact23793RawTerms
def rightRaw : List Term := Proof.Events093.exact23836RawTerms
def group : MergeGroup := .operator 23793 23836
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23793) (leftOrdinal := 0)
    (rightResult := 23836) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23840

namespace LeftMerge23857
def owner : Owner := ⟨.program ⟨257⟩, ⟨51365⟩⟩
def mergeEvent : Nat := 23857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51957⟩⟩] } }
def rhsRaw : List Term := Proof.Events093.exact23854RawTerms
def group : MergeGroup := .relation 23856
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23856) (rhsResult := 23854)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51362⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23855 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51362⟩⟩]⟩) (none) 23854) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51957⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], [⟨.program ⟨257⟩, ⟨51957⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23857

namespace LeftMerge23858
def owner : Owner := ⟨.program ⟨257⟩, ⟨51365⟩⟩
def mergeEvent : Nat := 23858
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩] } }
def rhsRaw : List Term := Proof.Events093.exact23854RawTerms
def group : MergeGroup := .relation 23856
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23856) (rhsResult := 23854)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51362⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23855 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51362⟩⟩]⟩) (none) 23854) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23858

namespace LeftMerge23859
def owner : Owner := ⟨.program ⟨257⟩, ⟨51365⟩⟩
def mergeEvent : Nat := 23859
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events093.exact23854RawTerms
def group : MergeGroup := .relation 23856
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23856) (rhsResult := 23854)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51362⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23855 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51362⟩⟩]⟩) (none) 23854) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23859

namespace LeftMerge23860
def owner : Owner := ⟨.program ⟨257⟩, ⟨51365⟩⟩
def mergeEvent : Nat := 23860
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events093.exact23854RawTerms
def group : MergeGroup := .relation 23856
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23856) (rhsResult := 23854)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51362⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23855 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51362⟩⟩]⟩) (none) 23854) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23860

namespace LeftMerge23865
def owner : Owner := ⟨.program ⟨257⟩, ⟨52425⟩⟩
def mergeEvent : Nat := 23865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51957⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact23861RawTerms
def rightRaw : List Term := Proof.Events092.exact23675RawTerms
def group : MergeGroup := .operator 23861 23675
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23861) (leftOrdinal := 2)
    (rightResult := 23675) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51957⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51957⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨24426⟩⟩, ⟨.program ⟨257⟩, ⟨50311⟩⟩], [⟨.program ⟨257⟩, ⟨51957⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23865

namespace LeftMerge23866
def owner : Owner := ⟨.program ⟨257⟩, ⟨52425⟩⟩
def mergeEvent : Nat := 23866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact23861RawTerms
def rightRaw : List Term := Proof.Events092.exact23675RawTerms
def group : MergeGroup := .operator 23861 23675
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23861) (leftOrdinal := 1)
    (rightResult := 23675) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52423⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23866

namespace LeftMerge23874
def owner : Owner := ⟨.program ⟨257⟩, ⟨52684⟩⟩
def mergeEvent : Nat := 23874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52682⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact23868RawTerms
def rightRaw : List Term := Proof.Events092.exact23572RawTerms
def group : MergeGroup := .operator 23868 23572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23868) (leftOrdinal := 1)
    (rightResult := 23572) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52682⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52682⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23874

namespace LeftMerge23876
def owner : Owner := ⟨.program ⟨257⟩, ⟨52684⟩⟩
def mergeEvent : Nat := 23876
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52083⟩⟩] } }
def rhsRaw : List Term := Proof.Events092.exact23569RawTerms
def group : MergeGroup := .relation 23875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23875) (rhsResult := 23569)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52682⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52682⟩⟩) ⟨52083⟩ 23569) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52083⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨52083⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23876

namespace LeftMerge23877
def owner : Owner := ⟨.program ⟨257⟩, ⟨52684⟩⟩
def mergeEvent : Nat := 23877
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52682⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact23868RawTerms
def rightRaw : List Term := Proof.Events092.exact23572RawTerms
def group : MergeGroup := .operator 23868 23572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23868) (leftOrdinal := 0)
    (rightResult := 23572) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52682⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52682⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23877

namespace LeftMerge23891
def owner : Owner := ⟨.program ⟨257⟩, ⟨51585⟩⟩
def mergeEvent : Nat := 23891
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51582⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events093.exact23885RawTerms
def group : MergeGroup := .operator 17169 23885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 23885) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51582⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51582⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23891

namespace LeftMerge24012
def owner : Owner := ⟨.program ⟨257⟩, ⟨52332⟩⟩
def mergeEvent : Nat := 24012
def frameStart : Nat := 23946
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact24008RawTerms
def rightRaw : List Term := Proof.Events093.exact24006RawTerms
def group : MergeGroup := .operator 24008 24006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24008) (leftOrdinal := 0)
    (rightResult := 24006) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24012

namespace LeftMerge24024
def owner : Owner := ⟨.program ⟨257⟩, ⟨52683⟩⟩
def mergeEvent : Nat := 24024
def frameStart : Nat := 23946
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52682⟩⟩] } }
def leftRaw : List Term := Proof.Events093.exact24020RawTerms
def rightRaw : List Term := Proof.Events093.exact23997RawTerms
def group : MergeGroup := .operator 24020 23997
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24020) (leftOrdinal := 1)
    (rightResult := 23997) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50818⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52682⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52682⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24024

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
