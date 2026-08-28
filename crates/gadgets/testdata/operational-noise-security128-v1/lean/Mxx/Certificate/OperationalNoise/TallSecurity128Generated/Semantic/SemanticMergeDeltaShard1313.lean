import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge213862
def owner : Owner := ⟨.program ⟨257⟩, ⟨50552⟩⟩
def mergeEvent : Nat := 213862
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events835.exact213853RawTerms
def rightRaw : List Term := Proof.Events092.exact23623RawTerms
def group : MergeGroup := .operator 213853 23623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213853) (leftOrdinal := 0)
    (rightResult := 23623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213862

namespace LeftMerge213867
def owner : Owner := ⟨.program ⟨257⟩, ⟨50553⟩⟩
def mergeEvent : Nat := 213867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }
def leftRaw : List Term := Proof.Events835.exact213863RawTerms
def rightRaw : List Term := Proof.Events835.exact213833RawTerms
def group : MergeGroup := .operator 213863 213833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213863) (leftOrdinal := 1)
    (rightResult := 213833) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213867

namespace LeftMerge213875
def owner : Owner := ⟨.program ⟨257⟩, ⟨52520⟩⟩
def mergeEvent : Nat := 213875
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩] } }
def leftRaw : List Term := Proof.Events835.exact213869RawTerms
def rightRaw : List Term := Proof.Events835.exact213805RawTerms
def group : MergeGroup := .operator 213869 213805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213869) (leftOrdinal := 1)
    (rightResult := 213805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52519⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213875

namespace LeftMerge213877
def owner : Owner := ⟨.program ⟨257⟩, ⟨52520⟩⟩
def mergeEvent : Nat := 213877
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52009⟩⟩] } }
def rhsRaw : List Term := Proof.Events835.exact213802RawTerms
def group : MergeGroup := .relation 213876
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213876) (rhsResult := 213802)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52519⟩⟩) ⟨52009⟩ 213802) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52009⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213877

namespace LeftMerge213878
def owner : Owner := ⟨.program ⟨257⟩, ⟨52520⟩⟩
def mergeEvent : Nat := 213878
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩] } }
def leftRaw : List Term := Proof.Events835.exact213869RawTerms
def rightRaw : List Term := Proof.Events835.exact213805RawTerms
def group : MergeGroup := .operator 213869 213805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213869) (leftOrdinal := 0)
    (rightResult := 213805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52519⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213878

namespace LeftMerge213892
def owner : Owner := ⟨.program ⟨257⟩, ⟨51452⟩⟩
def mergeEvent : Nat := 213892
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51449⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events835.exact213886RawTerms
def group : MergeGroup := .operator 207620 213886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 213886) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51449⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51449⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213892

namespace LeftMerge213971
def owner : Owner := ⟨.program ⟨257⟩, ⟨50546⟩⟩
def mergeEvent : Nat := 213971
def frameStart : Nat := 213941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events835.exact213967RawTerms
def rightRaw : List Term := Proof.Events835.exact213964RawTerms
def group : MergeGroup := .operator 213967 213964
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213967) (leftOrdinal := 0)
    (rightResult := 213964) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24530⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213971

namespace LeftMerge214001
def owner : Owner := ⟨.program ⟨257⟩, ⟨52288⟩⟩
def mergeEvent : Nat := 214001
def frameStart : Nat := 213941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events835.exact213997RawTerms
def rightRaw : List Term := Proof.Events835.exact213995RawTerms
def group : MergeGroup := .operator 213997 213995
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213997) (leftOrdinal := 0)
    (rightResult := 213995) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214001

namespace LeftMerge214024
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 214024
def frameStart : Nat := 213941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events836.exact214020RawTerms
def rightRaw : List Term := Proof.Events836.exact214017RawTerms
def group : MergeGroup := .operator 214020 214017
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214020) (leftOrdinal := 0)
    (rightResult := 214017) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214024

namespace LeftMerge214033
def owner : Owner := ⟨.program ⟨257⟩, ⟨52522⟩⟩
def mergeEvent : Nat := 214033
def frameStart : Nat := 213941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩] } }
def leftRaw : List Term := Proof.Events836.exact214029RawTerms
def rightRaw : List Term := Proof.Events835.exact213986RawTerms
def group : MergeGroup := .operator 214029 213986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214029) (leftOrdinal := 0)
    (rightResult := 213986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52519⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214033

namespace LeftMerge214034
def owner : Owner := ⟨.program ⟨257⟩, ⟨52522⟩⟩
def mergeEvent : Nat := 214034
def frameStart : Nat := 213941
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩] } }
def leftRaw : List Term := Proof.Events836.exact214029RawTerms
def rightRaw : List Term := Proof.Events835.exact213986RawTerms
def group : MergeGroup := .operator 214029 213986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214029) (leftOrdinal := 1)
    (rightResult := 213986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52519⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214034

namespace LeftMerge214036
def owner : Owner := ⟨.program ⟨257⟩, ⟨52522⟩⟩
def mergeEvent : Nat := 214036
def frameStart : Nat := 213941
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52009⟩⟩] } }
def rhsRaw : List Term := Proof.Events835.exact213983RawTerms
def group : MergeGroup := .relation 214035
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214035) (rhsResult := 213983)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52519⟩⟩) ⟨52009⟩ 213983) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52009⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214036

namespace LeftMerge214044
def owner : Owner := ⟨.program ⟨257⟩, ⟨50890⟩⟩
def mergeEvent : Nat := 214044
def frameStart : Nat := 213941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events835.exact213997RawTerms
def rightRaw : List Term := Proof.Events836.exact214040RawTerms
def group : MergeGroup := .operator 213997 214040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213997) (leftOrdinal := 0)
    (rightResult := 214040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50888⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214044

namespace LeftMerge214061
def owner : Owner := ⟨.program ⟨257⟩, ⟨51452⟩⟩
def mergeEvent : Nat := 214061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events836.exact214058RawTerms
def group : MergeGroup := .relation 214060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214060) (rhsResult := 214058)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214059 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51449⟩⟩]⟩) (none) 214058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214061

namespace LeftMerge214062
def owner : Owner := ⟨.program ⟨257⟩, ⟨51452⟩⟩
def mergeEvent : Nat := 214062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩] } }
def rhsRaw : List Term := Proof.Events836.exact214058RawTerms
def group : MergeGroup := .relation 214060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214060) (rhsResult := 214058)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214059 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51449⟩⟩]⟩) (none) 214058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214062

namespace LeftMerge214063
def owner : Owner := ⟨.program ⟨257⟩, ⟨51452⟩⟩
def mergeEvent : Nat := 214063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52009⟩⟩] } }
def rhsRaw : List Term := Proof.Events836.exact214058RawTerms
def group : MergeGroup := .relation 214060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214060) (rhsResult := 214058)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214059 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51449⟩⟩]⟩) (none) 214058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52009⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214063

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
