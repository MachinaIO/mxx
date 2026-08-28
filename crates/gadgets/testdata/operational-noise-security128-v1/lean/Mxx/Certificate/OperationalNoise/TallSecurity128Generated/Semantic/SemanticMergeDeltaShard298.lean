import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge52038
def owner : Owner := ⟨.program ⟨257⟩, ⟨58568⟩⟩
def mergeEvent : Nat := 52038
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58017⟩⟩] } }
def rhsRaw : List Term := Proof.Events202.exact51963RawTerms
def group : MergeGroup := .relation 52037
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52037) (rhsResult := 51963)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58567⟩⟩) ⟨58017⟩ 51963) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58017⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨58017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52038

namespace LeftMerge52039
def owner : Owner := ⟨.program ⟨257⟩, ⟨58568⟩⟩
def mergeEvent : Nat := 52039
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52030RawTerms
def rightRaw : List Term := Proof.Events202.exact51966RawTerms
def group : MergeGroup := .operator 52030 51966
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52030) (leftOrdinal := 0)
    (rightResult := 51966) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58567⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52039

namespace LeftMerge52053
def owner : Owner := ⟨.program ⟨257⟩, ⟨57492⟩⟩
def mergeEvent : Nat := 52053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57489⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events203.exact52047RawTerms
def group : MergeGroup := .operator 46745 52047
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 52047) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57489⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57489⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52053

namespace LeftMerge52132
def owner : Owner := ⟨.program ⟨257⟩, ⟨56722⟩⟩
def mergeEvent : Nat := 52132
def frameStart : Nat := 52102
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events203.exact52128RawTerms
def rightRaw : List Term := Proof.Events203.exact52125RawTerms
def group : MergeGroup := .operator 52128 52125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52128) (leftOrdinal := 0)
    (rightResult := 52125) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25106⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52132

namespace LeftMerge52162
def owner : Owner := ⟨.program ⟨257⟩, ⟨58280⟩⟩
def mergeEvent : Nat := 52162
def frameStart : Nat := 52102
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52158RawTerms
def rightRaw : List Term := Proof.Events203.exact52156RawTerms
def group : MergeGroup := .operator 52158 52156
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52158) (leftOrdinal := 0)
    (rightResult := 52156) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52162

namespace LeftMerge52185
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 52185
def frameStart : Nat := 52102
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52181RawTerms
def rightRaw : List Term := Proof.Events203.exact52178RawTerms
def group : MergeGroup := .operator 52181 52178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52181) (leftOrdinal := 0)
    (rightResult := 52178) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52185

namespace LeftMerge52194
def owner : Owner := ⟨.program ⟨257⟩, ⟨58570⟩⟩
def mergeEvent : Nat := 52194
def frameStart : Nat := 52102
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52190RawTerms
def rightRaw : List Term := Proof.Events203.exact52147RawTerms
def group : MergeGroup := .operator 52190 52147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52190) (leftOrdinal := 0)
    (rightResult := 52147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58567⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52194

namespace LeftMerge52195
def owner : Owner := ⟨.program ⟨257⟩, ⟨58570⟩⟩
def mergeEvent : Nat := 52195
def frameStart : Nat := 52102
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52190RawTerms
def rightRaw : List Term := Proof.Events203.exact52147RawTerms
def group : MergeGroup := .operator 52190 52147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52190) (leftOrdinal := 1)
    (rightResult := 52147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58567⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52195

namespace LeftMerge52197
def owner : Owner := ⟨.program ⟨257⟩, ⟨58570⟩⟩
def mergeEvent : Nat := 52197
def frameStart : Nat := 52102
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58017⟩⟩] } }
def rhsRaw : List Term := Proof.Events203.exact52144RawTerms
def group : MergeGroup := .relation 52196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52196) (rhsResult := 52144)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58567⟩⟩) ⟨58017⟩ 52144) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58017⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨58017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52197

namespace LeftMerge52205
def owner : Owner := ⟨.program ⟨257⟩, ⟨56914⟩⟩
def mergeEvent : Nat := 52205
def frameStart : Nat := 52102
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events203.exact52158RawTerms
def rightRaw : List Term := Proof.Events203.exact52201RawTerms
def group : MergeGroup := .operator 52158 52201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52158) (leftOrdinal := 0)
    (rightResult := 52201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56912⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52205

namespace LeftMerge52222
def owner : Owner := ⟨.program ⟨257⟩, ⟨57492⟩⟩
def mergeEvent : Nat := 52222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events203.exact52219RawTerms
def group : MergeGroup := .relation 52221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52221) (rhsResult := 52219)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57489⟩⟩]⟩) (none) 52219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52222

namespace LeftMerge52223
def owner : Owner := ⟨.program ⟨257⟩, ⟨57492⟩⟩
def mergeEvent : Nat := 52223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩] } }
def rhsRaw : List Term := Proof.Events203.exact52219RawTerms
def group : MergeGroup := .relation 52221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52221) (rhsResult := 52219)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57489⟩⟩]⟩) (none) 52219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52223

namespace LeftMerge52224
def owner : Owner := ⟨.program ⟨257⟩, ⟨57492⟩⟩
def mergeEvent : Nat := 52224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58017⟩⟩] } }
def rhsRaw : List Term := Proof.Events203.exact52219RawTerms
def group : MergeGroup := .relation 52221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52221) (rhsResult := 52219)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57489⟩⟩]⟩) (none) 52219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58017⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨58017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52224

namespace LeftMerge52225
def owner : Owner := ⟨.program ⟨257⟩, ⟨57492⟩⟩
def mergeEvent : Nat := 52225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events203.exact52219RawTerms
def group : MergeGroup := .relation 52221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52221) (rhsResult := 52219)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 52220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57489⟩⟩]⟩) (none) 52219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨56912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52225

namespace LeftMerge52230
def owner : Owner := ⟨.program ⟨257⟩, ⟨58569⟩⟩
def mergeEvent : Nat := 52230
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58017⟩⟩] } }
def leftRaw : List Term := Proof.Events204.exact52226RawTerms
def rightRaw : List Term := Proof.Events203.exact52040RawTerms
def group : MergeGroup := .operator 52226 52040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52226) (leftOrdinal := 2)
    (rightResult := 52040) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58017⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25106⟩⟩, ⟨.program ⟨257⟩, ⟨56721⟩⟩], [⟨.program ⟨257⟩, ⟨58017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52230

namespace LeftMerge52231
def owner : Owner := ⟨.program ⟨257⟩, ⟨58569⟩⟩
def mergeEvent : Nat := 52231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩] } }
def leftRaw : List Term := Proof.Events204.exact52226RawTerms
def rightRaw : List Term := Proof.Events203.exact52040RawTerms
def group : MergeGroup := .operator 52226 52040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52226) (leftOrdinal := 1)
    (rightResult := 52040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58567⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52231

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
