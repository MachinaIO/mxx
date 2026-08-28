import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge60892
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60892
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 4) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60892

namespace LeftMerge60893
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60893
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60893

namespace LeftMerge60894
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60894
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60894

namespace LeftMerge60895
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60895
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60895

namespace LeftMerge60896
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60896
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60896

namespace LeftMerge60897
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60897
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18173⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 34) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18173⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18173⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60897

namespace LeftMerge60898
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60898
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17088⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 30) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17088⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17088⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60898

namespace LeftMerge60899
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60899
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16801⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 29) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16801⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16801⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60899

namespace LeftMerge60900
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60900
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16682⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 28) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16682⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16682⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60900

namespace LeftMerge60901
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18208⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 35) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18208⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18208⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60901

namespace LeftMerge60902
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17907⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 33) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17907⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17907⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60902

namespace LeftMerge60903
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17123⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 31) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17123⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17123⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60903

namespace LeftMerge60904
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60904
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16311⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 27) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16311⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16311⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60904

namespace LeftMerge60905
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60905
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 36) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60905

namespace LeftMerge60906
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60906
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16108⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 26) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16108⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60906

namespace LeftMerge60907
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 60907
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events237.exact60875RawTerms
def group : MergeGroup := .relation 60877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60877) (rhsResult := 60875)
    (sourceTermOrdinal := 25) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60876 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩) (none) 60875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15989⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15989⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60907

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
