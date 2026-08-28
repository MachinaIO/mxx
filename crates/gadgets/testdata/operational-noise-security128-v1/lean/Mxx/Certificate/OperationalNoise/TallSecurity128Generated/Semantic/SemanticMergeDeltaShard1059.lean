import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge173874
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 5) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173874

namespace LeftMerge173875
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173875
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 4) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173875

namespace LeftMerge173876
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173876
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173876

namespace LeftMerge173877
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173877
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173877

namespace LeftMerge173878
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173878
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173878

namespace LeftMerge173879
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173879
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173879

namespace LeftMerge173880
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173880
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48415⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 30) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48415⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173880

namespace LeftMerge173881
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173881
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45735⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 29) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45735⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173881

namespace LeftMerge173882
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173882
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨43051⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 28) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43051⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173882

namespace LeftMerge173883
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 27) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173883

namespace LeftMerge173884
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173884
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 26) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37695⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173884

namespace LeftMerge173885
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173885
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 25) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35015⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173885

namespace LeftMerge173886
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173886
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 23) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29351⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173886

namespace LeftMerge173887
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173887
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 22) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26671⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173887

namespace LeftMerge173888
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173888
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 36) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66881⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173888

namespace LeftMerge173889
def owner : Owner := ⟨.program ⟨257⟩, ⟨68413⟩⟩
def mergeEvent : Nat := 173889
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events679.exact173858RawTerms
def group : MergeGroup := .relation 173860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173860) (rhsResult := 173858)
    (sourceTermOrdinal := 35) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 173859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) (none) 173858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63157⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173889

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
