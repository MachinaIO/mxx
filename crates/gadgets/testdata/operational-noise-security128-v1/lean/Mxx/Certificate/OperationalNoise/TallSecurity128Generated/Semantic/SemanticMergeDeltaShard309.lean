import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge54125
def owner : Owner := ⟨.program ⟨257⟩, ⟨23530⟩⟩
def mergeEvent : Nat := 54125
def frameStart : Nat := 54030
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22977⟩⟩] } }
def rhsRaw : List Term := Proof.Events211.exact54072RawTerms
def group : MergeGroup := .relation 54124
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54124) (rhsResult := 54072)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23527⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23527⟩⟩) ⟨22977⟩ 54072) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22977⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], [⟨.program ⟨257⟩, ⟨22977⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54125

namespace LeftMerge54133
def owner : Owner := ⟨.program ⟨257⟩, ⟨21874⟩⟩
def mergeEvent : Nat := 54133
def frameStart : Nat := 54030
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events211.exact54086RawTerms
def rightRaw : List Term := Proof.Events211.exact54129RawTerms
def group : MergeGroup := .operator 54086 54129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54086) (leftOrdinal := 0)
    (rightResult := 54129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54133

namespace LeftMerge54150
def owner : Owner := ⟨.program ⟨257⟩, ⟨22452⟩⟩
def mergeEvent : Nat := 54150
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events211.exact54147RawTerms
def group : MergeGroup := .relation 54149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54149) (rhsResult := 54147)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54148 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22449⟩⟩]⟩) (none) 54147) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54150

namespace LeftMerge54151
def owner : Owner := ⟨.program ⟨257⟩, ⟨22452⟩⟩
def mergeEvent : Nat := 54151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23527⟩⟩] } }
def rhsRaw : List Term := Proof.Events211.exact54147RawTerms
def group : MergeGroup := .relation 54149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54149) (rhsResult := 54147)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54148 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22449⟩⟩]⟩) (none) 54147) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23527⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23527⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54151

namespace LeftMerge54152
def owner : Owner := ⟨.program ⟨257⟩, ⟨22452⟩⟩
def mergeEvent : Nat := 54152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22977⟩⟩] } }
def rhsRaw : List Term := Proof.Events211.exact54147RawTerms
def group : MergeGroup := .relation 54149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54149) (rhsResult := 54147)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54148 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22449⟩⟩]⟩) (none) 54147) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22977⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], [⟨.program ⟨257⟩, ⟨22977⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54152

namespace LeftMerge54153
def owner : Owner := ⟨.program ⟨257⟩, ⟨22452⟩⟩
def mergeEvent : Nat := 54153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events211.exact54147RawTerms
def group : MergeGroup := .relation 54149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54149) (rhsResult := 54147)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54148 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22449⟩⟩]⟩) (none) 54147) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54153

namespace LeftMerge54158
def owner : Owner := ⟨.program ⟨257⟩, ⟨23529⟩⟩
def mergeEvent : Nat := 54158
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22977⟩⟩] } }
def leftRaw : List Term := Proof.Events211.exact54154RawTerms
def rightRaw : List Term := Proof.Events210.exact53968RawTerms
def group : MergeGroup := .operator 54154 53968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54154) (leftOrdinal := 2)
    (rightResult := 53968) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22977⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22977⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21221⟩⟩, ⟨.program ⟨257⟩, ⟨21686⟩⟩], [⟨.program ⟨257⟩, ⟨22977⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54158

namespace LeftMerge54159
def owner : Owner := ⟨.program ⟨257⟩, ⟨23529⟩⟩
def mergeEvent : Nat := 54159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23527⟩⟩] } }
def leftRaw : List Term := Proof.Events211.exact54154RawTerms
def rightRaw : List Term := Proof.Events210.exact53968RawTerms
def group : MergeGroup := .operator 54154 53968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54154) (leftOrdinal := 1)
    (rightResult := 53968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23527⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23527⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23527⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54159

namespace LeftMerge54167
def owner : Owner := ⟨.program ⟨257⟩, ⟨24122⟩⟩
def mergeEvent : Nat := 54167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24120⟩⟩] } }
def leftRaw : List Term := Proof.Events211.exact54161RawTerms
def rightRaw : List Term := Proof.Events210.exact53884RawTerms
def group : MergeGroup := .operator 54161 53884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54161) (leftOrdinal := 0)
    (rightResult := 53884) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨24120⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24120⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54167

namespace LeftMerge54168
def owner : Owner := ⟨.program ⟨257⟩, ⟨24122⟩⟩
def mergeEvent : Nat := 54168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24120⟩⟩] } }
def leftRaw : List Term := Proof.Events211.exact54161RawTerms
def rightRaw : List Term := Proof.Events210.exact53884RawTerms
def group : MergeGroup := .operator 54161 53884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54161) (leftOrdinal := 1)
    (rightResult := 53884) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨24120⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24120⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54168

namespace LeftMerge54170
def owner : Owner := ⟨.program ⟨257⟩, ⟨24122⟩⟩
def mergeEvent : Nat := 54170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23153⟩⟩] } }
def rhsRaw : List Term := Proof.Events210.exact53881RawTerms
def group : MergeGroup := .relation 54169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54169) (rhsResult := 53881)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24120⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨24120⟩⟩) ⟨23153⟩ 53881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23153⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54170

namespace LeftMerge54184
def owner : Owner := ⟨.program ⟨257⟩, ⟨22839⟩⟩
def mergeEvent : Nat := 54184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22836⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events211.exact54178RawTerms
def group : MergeGroup := .operator 46745 54178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 54178) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22836⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54184

namespace LeftMerge54305
def owner : Owner := ⟨.program ⟨257⟩, ⟨23320⟩⟩
def mergeEvent : Nat := 54305
def frameStart : Nat := 54239
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54301RawTerms
def rightRaw : List Term := Proof.Events212.exact54299RawTerms
def group : MergeGroup := .operator 54301 54299
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54301) (leftOrdinal := 0)
    (rightResult := 54299) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54305

namespace LeftMerge54317
def owner : Owner := ⟨.program ⟨257⟩, ⟨24121⟩⟩
def mergeEvent : Nat := 54317
def frameStart : Nat := 54239
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24120⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54313RawTerms
def rightRaw : List Term := Proof.Events212.exact54290RawTerms
def group : MergeGroup := .operator 54313 54290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54313) (leftOrdinal := 0)
    (rightResult := 54290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨24120⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24120⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54317

namespace LeftMerge54318
def owner : Owner := ⟨.program ⟨257⟩, ⟨24121⟩⟩
def mergeEvent : Nat := 54318
def frameStart : Nat := 54239
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24120⟩⟩] } }
def leftRaw : List Term := Proof.Events212.exact54313RawTerms
def rightRaw : List Term := Proof.Events212.exact54290RawTerms
def group : MergeGroup := .operator 54313 54290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54313) (leftOrdinal := 1)
    (rightResult := 54290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨24120⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24120⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54318

namespace LeftMerge54320
def owner : Owner := ⟨.program ⟨257⟩, ⟨24121⟩⟩
def mergeEvent : Nat := 54320
def frameStart : Nat := 54239
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23153⟩⟩] } }
def rhsRaw : List Term := Proof.Events212.exact54287RawTerms
def group : MergeGroup := .relation 54319
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54319) (rhsResult := 54287)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24120⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨24120⟩⟩) ⟨23153⟩ 54287) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23153⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54320

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
