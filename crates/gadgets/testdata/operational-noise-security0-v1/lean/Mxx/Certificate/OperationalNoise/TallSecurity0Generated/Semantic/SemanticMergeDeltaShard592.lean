import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge95916
def owner : Owner := ⟨.program ⟨214⟩, ⟨20024⟩⟩
def mergeEvent : Nat := 95916
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩] } }
def rhsRaw : List Term := Proof.Events374.exact95912RawTerms
def group : MergeGroup := .relation 95914
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95914) (rhsResult := 95912)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20021⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95913 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20021⟩⟩]⟩) (none) 95912) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95916

namespace LeftMerge95917
def owner : Owner := ⟨.program ⟨214⟩, ⟨20024⟩⟩
def mergeEvent : Nat := 95917
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23284⟩⟩] } }
def rhsRaw : List Term := Proof.Events374.exact95912RawTerms
def group : MergeGroup := .relation 95914
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95914) (rhsResult := 95912)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20021⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95913 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20021⟩⟩]⟩) (none) 95912) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23284⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨23284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95917

namespace LeftMerge95918
def owner : Owner := ⟨.program ⟨214⟩, ⟨20024⟩⟩
def mergeEvent : Nat := 95918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events374.exact95912RawTerms
def group : MergeGroup := .relation 95914
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95914) (rhsResult := 95912)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20021⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95913 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20021⟩⟩]⟩) (none) 95912) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95918

namespace LeftMerge95923
def owner : Owner := ⟨.program ⟨214⟩, ⟨25516⟩⟩
def mergeEvent : Nat := 95923
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23284⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95919RawTerms
def rightRaw : List Term := Proof.Events374.exact95757RawTerms
def group : MergeGroup := .operator 95919 95757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95919) (leftOrdinal := 2)
    (rightResult := 95757) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23284⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23284⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨23284⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95923

namespace LeftMerge95924
def owner : Owner := ⟨.program ⟨214⟩, ⟨25516⟩⟩
def mergeEvent : Nat := 95924
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95919RawTerms
def rightRaw : List Term := Proof.Events374.exact95757RawTerms
def group : MergeGroup := .operator 95919 95757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95919) (leftOrdinal := 1)
    (rightResult := 95757) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95924

namespace LeftMerge95932
def owner : Owner := ⟨.program ⟨214⟩, ⟨29352⟩⟩
def mergeEvent : Nat := 95932
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95926RawTerms
def rightRaw : List Term := Proof.Events373.exact95673RawTerms
def group : MergeGroup := .operator 95926 95673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95926) (leftOrdinal := 0)
    (rightResult := 95673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29350⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95932

namespace LeftMerge95933
def owner : Owner := ⟨.program ⟨214⟩, ⟨29352⟩⟩
def mergeEvent : Nat := 95933
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95926RawTerms
def rightRaw : List Term := Proof.Events373.exact95673RawTerms
def group : MergeGroup := .operator 95926 95673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95926) (leftOrdinal := 1)
    (rightResult := 95673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29350⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95933

namespace LeftMerge95935
def owner : Owner := ⟨.program ⟨214⟩, ⟨29352⟩⟩
def mergeEvent : Nat := 95935
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24594⟩⟩] } }
def rhsRaw : List Term := Proof.Events373.exact95670RawTerms
def group : MergeGroup := .relation 95934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95934) (rhsResult := 95670)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29350⟩⟩) ⟨24594⟩ 95670) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24594⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨24594⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95935

namespace LeftMerge95949
def owner : Owner := ⟨.program ⟨214⟩, ⟨22400⟩⟩
def mergeEvent : Nat := 95949
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22397⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events374.exact95943RawTerms
def group : MergeGroup := .operator 94462 95943
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 95943) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22397⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22397⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95949

namespace LeftMerge96046
def owner : Owner := ⟨.program ⟨214⟩, ⟨16702⟩⟩
def mergeEvent : Nat := 96046
def frameStart : Nat := 95992
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events375.exact96042RawTerms
def rightRaw : List Term := Proof.Events375.exact96040RawTerms
def group : MergeGroup := .operator 96042 96040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96042) (leftOrdinal := 0)
    (rightResult := 96040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96046

namespace LeftMerge96058
def owner : Owner := ⟨.program ⟨214⟩, ⟨29351⟩⟩
def mergeEvent : Nat := 96058
def frameStart : Nat := 95992
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩] } }
def leftRaw : List Term := Proof.Events375.exact96054RawTerms
def rightRaw : List Term := Proof.Events375.exact96031RawTerms
def group : MergeGroup := .operator 96054 96031
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96054) (leftOrdinal := 0)
    (rightResult := 96031) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29350⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96058

namespace LeftMerge96059
def owner : Owner := ⟨.program ⟨214⟩, ⟨29351⟩⟩
def mergeEvent : Nat := 96059
def frameStart : Nat := 95992
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩] } }
def leftRaw : List Term := Proof.Events375.exact96054RawTerms
def rightRaw : List Term := Proof.Events375.exact96031RawTerms
def group : MergeGroup := .operator 96054 96031
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96054) (leftOrdinal := 1)
    (rightResult := 96031) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29350⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96059

namespace LeftMerge96061
def owner : Owner := ⟨.program ⟨214⟩, ⟨29351⟩⟩
def mergeEvent : Nat := 96061
def frameStart : Nat := 95992
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24594⟩⟩] } }
def rhsRaw : List Term := Proof.Events375.exact96028RawTerms
def group : MergeGroup := .relation 96060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96060) (rhsResult := 96028)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29350⟩⟩) ⟨24594⟩ 96028) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24594⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨24594⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96061

namespace LeftMerge96069
def owner : Owner := ⟨.program ⟨214⟩, ⟨16673⟩⟩
def mergeEvent : Nat := 96069
def frameStart : Nat := 95992
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16672⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events375.exact96042RawTerms
def rightRaw : List Term := Proof.Events375.exact96065RawTerms
def group : MergeGroup := .operator 96042 96065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96042) (leftOrdinal := 0)
    (rightResult := 96065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16672⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96069

namespace LeftMerge96086
def owner : Owner := ⟨.program ⟨214⟩, ⟨22400⟩⟩
def mergeEvent : Nat := 96086
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }
def rhsRaw : List Term := Proof.Events375.exact96083RawTerms
def group : MergeGroup := .relation 96085
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96085) (rhsResult := 96083)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22397⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96084 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22397⟩⟩]⟩) (none) 96083) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96086

namespace LeftMerge96087
def owner : Owner := ⟨.program ⟨214⟩, ⟨22400⟩⟩
def mergeEvent : Nat := 96087
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩] } }
def rhsRaw : List Term := Proof.Events375.exact96083RawTerms
def group : MergeGroup := .relation 96085
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96085) (rhsResult := 96083)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22397⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96084 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22397⟩⟩]⟩) (none) 96083) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29350⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96087

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
