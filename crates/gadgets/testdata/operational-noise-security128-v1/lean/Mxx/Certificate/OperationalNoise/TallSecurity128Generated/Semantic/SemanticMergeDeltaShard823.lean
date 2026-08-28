import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge135124
def owner : Owner := ⟨.program ⟨257⟩, ⟨46905⟩⟩
def mergeEvent : Nat := 135124
def frameStart : Nat := 135032
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩] } }
def leftRaw : List Term := Proof.Events527.exact135120RawTerms
def rightRaw : List Term := Proof.Events527.exact135077RawTerms
def group : MergeGroup := .operator 135120 135077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 135120) (leftOrdinal := 0)
    (rightResult := 135077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46902⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge135124

namespace LeftMerge135125
def owner : Owner := ⟨.program ⟨257⟩, ⟨46905⟩⟩
def mergeEvent : Nat := 135125
def frameStart : Nat := 135032
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩] } }
def leftRaw : List Term := Proof.Events527.exact135120RawTerms
def rightRaw : List Term := Proof.Events527.exact135077RawTerms
def group : MergeGroup := .operator 135120 135077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 135120) (leftOrdinal := 1)
    (rightResult := 135077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46902⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge135125

namespace LeftMerge135127
def owner : Owner := ⟨.program ⟨257⟩, ⟨46905⟩⟩
def mergeEvent : Nat := 135127
def frameStart : Nat := 135032
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46427⟩⟩] } }
def rhsRaw : List Term := Proof.Events527.exact135074RawTerms
def group : MergeGroup := .relation 135126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 135126) (rhsResult := 135074)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46902⟩⟩) ⟨46427⟩ 135074) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46427⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], [⟨.program ⟨257⟩, ⟨46427⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge135127

namespace LeftMerge135135
def owner : Owner := ⟨.program ⟨257⟩, ⟨45414⟩⟩
def mergeEvent : Nat := 135135
def frameStart : Nat := 135032
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events527.exact135088RawTerms
def rightRaw : List Term := Proof.Events527.exact135131RawTerms
def group : MergeGroup := .operator 135088 135131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 135088) (leftOrdinal := 0)
    (rightResult := 135131) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45412⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45412⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge135135

namespace LeftMerge135152
def owner : Owner := ⟨.program ⟨257⟩, ⟨45842⟩⟩
def mergeEvent : Nat := 135152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events527.exact135149RawTerms
def group : MergeGroup := .relation 135151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 135151) (rhsResult := 135149)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45839⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 135150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45839⟩⟩]⟩) (none) 135149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge135152

namespace LeftMerge135153
def owner : Owner := ⟨.program ⟨257⟩, ⟨45842⟩⟩
def mergeEvent : Nat := 135153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩] } }
def rhsRaw : List Term := Proof.Events527.exact135149RawTerms
def group : MergeGroup := .relation 135151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 135151) (rhsResult := 135149)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45839⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 135150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45839⟩⟩]⟩) (none) 135149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge135153

namespace LeftMerge135154
def owner : Owner := ⟨.program ⟨257⟩, ⟨45842⟩⟩
def mergeEvent : Nat := 135154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46427⟩⟩] } }
def rhsRaw : List Term := Proof.Events527.exact135149RawTerms
def group : MergeGroup := .relation 135151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 135151) (rhsResult := 135149)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45839⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 135150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45839⟩⟩]⟩) (none) 135149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46427⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], [⟨.program ⟨257⟩, ⟨46427⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge135154

namespace LeftMerge135155
def owner : Owner := ⟨.program ⟨257⟩, ⟨45842⟩⟩
def mergeEvent : Nat := 135155
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events527.exact135149RawTerms
def group : MergeGroup := .relation 135151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 135151) (rhsResult := 135149)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45839⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 135150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45839⟩⟩]⟩) (none) 135149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45412⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge135155

namespace LeftMerge135160
def owner : Owner := ⟨.program ⟨257⟩, ⟨46904⟩⟩
def mergeEvent : Nat := 135160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46427⟩⟩] } }
def leftRaw : List Term := Proof.Events527.exact135156RawTerms
def rightRaw : List Term := Proof.Events527.exact134970RawTerms
def group : MergeGroup := .operator 135156 134970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 135156) (leftOrdinal := 2)
    (rightResult := 134970) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46427⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46427⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], [⟨.program ⟨257⟩, ⟨46427⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge135160

namespace LeftMerge135161
def owner : Owner := ⟨.program ⟨257⟩, ⟨46904⟩⟩
def mergeEvent : Nat := 135161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩] } }
def leftRaw : List Term := Proof.Events527.exact135156RawTerms
def rightRaw : List Term := Proof.Events527.exact134970RawTerms
def group : MergeGroup := .operator 135156 134970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 135156) (leftOrdinal := 1)
    (rightResult := 134970) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46902⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge135161

namespace LeftMerge135169
def owner : Owner := ⟨.program ⟨257⟩, ⟨47176⟩⟩
def mergeEvent : Nat := 135169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47174⟩⟩] } }
def leftRaw : List Term := Proof.Events527.exact135163RawTerms
def rightRaw : List Term := Proof.Events526.exact134886RawTerms
def group : MergeGroup := .operator 135163 134886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 135163) (leftOrdinal := 0)
    (rightResult := 134886) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47174⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47174⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge135169

namespace LeftMerge135170
def owner : Owner := ⟨.program ⟨257⟩, ⟨47176⟩⟩
def mergeEvent : Nat := 135170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47174⟩⟩] } }
def leftRaw : List Term := Proof.Events527.exact135163RawTerms
def rightRaw : List Term := Proof.Events526.exact134886RawTerms
def group : MergeGroup := .operator 135163 134886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 135163) (leftOrdinal := 1)
    (rightResult := 134886) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47174⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45412⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47174⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge135170

namespace LeftMerge135172
def owner : Owner := ⟨.program ⟨257⟩, ⟨47176⟩⟩
def mergeEvent : Nat := 135172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46558⟩⟩] } }
def rhsRaw : List Term := Proof.Events526.exact134883RawTerms
def group : MergeGroup := .relation 135171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 135171) (rhsResult := 134883)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45412⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47174⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47174⟩⟩) ⟨46558⟩ 134883) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46558⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45412⟩⟩], [⟨.program ⟨257⟩, ⟨46558⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge135172

namespace LeftMerge135186
def owner : Owner := ⟨.program ⟨257⟩, ⟨46079⟩⟩
def mergeEvent : Nat := 135186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46076⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events528.exact135180RawTerms
def group : MergeGroup := .operator 134495 135180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 135180) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46076⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46076⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge135186

namespace LeftMerge135307
def owner : Owner := ⟨.program ⟨257⟩, ⟨46800⟩⟩
def mergeEvent : Nat := 135307
def frameStart : Nat := 135241
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events528.exact135303RawTerms
def rightRaw : List Term := Proof.Events528.exact135301RawTerms
def group : MergeGroup := .operator 135303 135301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 135303) (leftOrdinal := 0)
    (rightResult := 135301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45412⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45412⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge135307

namespace LeftMerge135319
def owner : Owner := ⟨.program ⟨257⟩, ⟨47175⟩⟩
def mergeEvent : Nat := 135319
def frameStart : Nat := 135241
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47174⟩⟩] } }
def leftRaw : List Term := Proof.Events528.exact135315RawTerms
def rightRaw : List Term := Proof.Events528.exact135292RawTerms
def group : MergeGroup := .operator 135315 135292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 135315) (leftOrdinal := 0)
    (rightResult := 135292) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47174⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47174⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge135319

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
