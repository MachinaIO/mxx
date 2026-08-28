import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge212914
def owner : Owner := ⟨.program ⟨257⟩, ⟨58480⟩⟩
def mergeEvent : Nat := 212914
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩] } }
def leftRaw : List Term := Proof.Events831.exact212905RawTerms
def rightRaw : List Term := Proof.Events831.exact212841RawTerms
def group : MergeGroup := .operator 212905 212841
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 212905) (leftOrdinal := 0)
    (rightResult := 212841) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58479⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge212914

namespace LeftMerge212928
def owner : Owner := ⟨.program ⟨257⟩, ⟨57412⟩⟩
def mergeEvent : Nat := 212928
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57409⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events831.exact212922RawTerms
def group : MergeGroup := .operator 207620 212922
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 212922) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57409⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57409⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge212928

namespace LeftMerge213007
def owner : Owner := ⟨.program ⟨257⟩, ⟨56506⟩⟩
def mergeEvent : Nat := 213007
def frameStart : Nat := 212977
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events832.exact213003RawTerms
def rightRaw : List Term := Proof.Events832.exact213000RawTerms
def group : MergeGroup := .operator 213003 213000
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213003) (leftOrdinal := 0)
    (rightResult := 213000) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25010⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213007

namespace LeftMerge213037
def owner : Owner := ⟨.program ⟨257⟩, ⟨58248⟩⟩
def mergeEvent : Nat := 213037
def frameStart : Nat := 212977
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events832.exact213033RawTerms
def rightRaw : List Term := Proof.Events832.exact213031RawTerms
def group : MergeGroup := .operator 213033 213031
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213033) (leftOrdinal := 0)
    (rightResult := 213031) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213037

namespace LeftMerge213060
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 213060
def frameStart : Nat := 212977
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events832.exact213056RawTerms
def rightRaw : List Term := Proof.Events832.exact213053RawTerms
def group : MergeGroup := .operator 213056 213053
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213056) (leftOrdinal := 0)
    (rightResult := 213053) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213060

namespace LeftMerge213069
def owner : Owner := ⟨.program ⟨257⟩, ⟨58482⟩⟩
def mergeEvent : Nat := 213069
def frameStart : Nat := 212977
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩] } }
def leftRaw : List Term := Proof.Events832.exact213065RawTerms
def rightRaw : List Term := Proof.Events832.exact213022RawTerms
def group : MergeGroup := .operator 213065 213022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213065) (leftOrdinal := 0)
    (rightResult := 213022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58479⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213069

namespace LeftMerge213070
def owner : Owner := ⟨.program ⟨257⟩, ⟨58482⟩⟩
def mergeEvent : Nat := 213070
def frameStart : Nat := 212977
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩] } }
def leftRaw : List Term := Proof.Events832.exact213065RawTerms
def rightRaw : List Term := Proof.Events832.exact213022RawTerms
def group : MergeGroup := .operator 213065 213022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213065) (leftOrdinal := 1)
    (rightResult := 213022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58479⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213070

namespace LeftMerge213072
def owner : Owner := ⟨.program ⟨257⟩, ⟨58482⟩⟩
def mergeEvent : Nat := 213072
def frameStart : Nat := 212977
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57969⟩⟩] } }
def rhsRaw : List Term := Proof.Events832.exact213019RawTerms
def group : MergeGroup := .relation 213071
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213071) (rhsResult := 213019)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58479⟩⟩) ⟨57969⟩ 213019) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57969⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], [⟨.program ⟨257⟩, ⟨57969⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213072

namespace LeftMerge213080
def owner : Owner := ⟨.program ⟨257⟩, ⟨56850⟩⟩
def mergeEvent : Nat := 213080
def frameStart : Nat := 212977
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events832.exact213033RawTerms
def rightRaw : List Term := Proof.Events832.exact213076RawTerms
def group : MergeGroup := .operator 213033 213076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213033) (leftOrdinal := 0)
    (rightResult := 213076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56848⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213080

namespace LeftMerge213097
def owner : Owner := ⟨.program ⟨257⟩, ⟨57412⟩⟩
def mergeEvent : Nat := 213097
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events832.exact213094RawTerms
def group : MergeGroup := .relation 213096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213096) (rhsResult := 213094)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 213095 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57409⟩⟩]⟩) (none) 213094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213097

namespace LeftMerge213098
def owner : Owner := ⟨.program ⟨257⟩, ⟨57412⟩⟩
def mergeEvent : Nat := 213098
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩] } }
def rhsRaw : List Term := Proof.Events832.exact213094RawTerms
def group : MergeGroup := .relation 213096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213096) (rhsResult := 213094)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 213095 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57409⟩⟩]⟩) (none) 213094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213098

namespace LeftMerge213099
def owner : Owner := ⟨.program ⟨257⟩, ⟨57412⟩⟩
def mergeEvent : Nat := 213099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57969⟩⟩] } }
def rhsRaw : List Term := Proof.Events832.exact213094RawTerms
def group : MergeGroup := .relation 213096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213096) (rhsResult := 213094)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 213095 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57409⟩⟩]⟩) (none) 213094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57969⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], [⟨.program ⟨257⟩, ⟨57969⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213099

namespace LeftMerge213100
def owner : Owner := ⟨.program ⟨257⟩, ⟨57412⟩⟩
def mergeEvent : Nat := 213100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨56848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events832.exact213094RawTerms
def group : MergeGroup := .relation 213096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 213096) (rhsResult := 213094)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 213095 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57409⟩⟩]⟩) (none) 213094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨56848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213100

namespace LeftMerge213105
def owner : Owner := ⟨.program ⟨257⟩, ⟨58481⟩⟩
def mergeEvent : Nat := 213105
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57969⟩⟩] } }
def leftRaw : List Term := Proof.Events832.exact213101RawTerms
def rightRaw : List Term := Proof.Events831.exact212915RawTerms
def group : MergeGroup := .operator 213101 212915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213101) (leftOrdinal := 2)
    (rightResult := 212915) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57969⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57969⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25010⟩⟩, ⟨.program ⟨257⟩, ⟨56505⟩⟩], [⟨.program ⟨257⟩, ⟨57969⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge213105

namespace LeftMerge213106
def owner : Owner := ⟨.program ⟨257⟩, ⟨58481⟩⟩
def mergeEvent : Nat := 213106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩] } }
def leftRaw : List Term := Proof.Events832.exact213101RawTerms
def rightRaw : List Term := Proof.Events831.exact212915RawTerms
def group : MergeGroup := .operator 213101 212915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213101) (leftOrdinal := 1)
    (rightResult := 212915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58479⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213106

namespace LeftMerge213114
def owner : Owner := ⟨.program ⟨257⟩, ⟨58914⟩⟩
def mergeEvent : Nat := 213114
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58912⟩⟩] } }
def leftRaw : List Term := Proof.Events832.exact213108RawTerms
def rightRaw : List Term := Proof.Events831.exact212831RawTerms
def group : MergeGroup := .operator 213108 212831
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213108) (leftOrdinal := 0)
    (rightResult := 212831) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58912⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58912⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge213114

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
