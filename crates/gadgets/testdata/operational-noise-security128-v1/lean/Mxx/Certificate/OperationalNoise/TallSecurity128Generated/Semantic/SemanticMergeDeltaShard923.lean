import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge151224
def owner : Owner := ⟨.program ⟨257⟩, ⟨37842⟩⟩
def mergeEvent : Nat := 151224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38906⟩⟩] } }
def rhsRaw : List Term := Proof.Events590.exact151220RawTerms
def group : MergeGroup := .relation 151222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151222) (rhsResult := 151220)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37839⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 151221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37839⟩⟩]⟩) (none) 151220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38906⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38906⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151224

namespace LeftMerge151225
def owner : Owner := ⟨.program ⟨257⟩, ⟨37842⟩⟩
def mergeEvent : Nat := 151225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13836⟩⟩, ⟨.program ⟨257⟩, ⟨37042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38411⟩⟩] } }
def rhsRaw : List Term := Proof.Events590.exact151220RawTerms
def group : MergeGroup := .relation 151222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151222) (rhsResult := 151220)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37839⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 151221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37839⟩⟩]⟩) (none) 151220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13836⟩⟩, ⟨.program ⟨257⟩, ⟨37042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38411⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13836⟩⟩, ⟨.program ⟨257⟩, ⟨37042⟩⟩], [⟨.program ⟨257⟩, ⟨38411⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151225

namespace LeftMerge151226
def owner : Owner := ⟨.program ⟨257⟩, ⟨37842⟩⟩
def mergeEvent : Nat := 151226
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events590.exact151220RawTerms
def group : MergeGroup := .relation 151222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151222) (rhsResult := 151220)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37839⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 151221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37839⟩⟩]⟩) (none) 151220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151226

namespace LeftMerge151231
def owner : Owner := ⟨.program ⟨257⟩, ⟨38908⟩⟩
def mergeEvent : Nat := 151231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13836⟩⟩, ⟨.program ⟨257⟩, ⟨37042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38411⟩⟩] } }
def leftRaw : List Term := Proof.Events590.exact151227RawTerms
def rightRaw : List Term := Proof.Events590.exact151041RawTerms
def group : MergeGroup := .operator 151227 151041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151227) (leftOrdinal := 2)
    (rightResult := 151041) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13836⟩⟩, ⟨.program ⟨257⟩, ⟨37042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38411⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13836⟩⟩, ⟨.program ⟨257⟩, ⟨37042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38411⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13836⟩⟩, ⟨.program ⟨257⟩, ⟨37042⟩⟩], [⟨.program ⟨257⟩, ⟨38411⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151231

namespace LeftMerge151232
def owner : Owner := ⟨.program ⟨257⟩, ⟨38908⟩⟩
def mergeEvent : Nat := 151232
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38906⟩⟩] } }
def leftRaw : List Term := Proof.Events590.exact151227RawTerms
def rightRaw : List Term := Proof.Events590.exact151041RawTerms
def group : MergeGroup := .operator 151227 151041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151227) (leftOrdinal := 1)
    (rightResult := 151041) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38906⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38906⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38906⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151232

namespace LeftMerge151240
def owner : Owner := ⟨.program ⟨257⟩, ⟨39236⟩⟩
def mergeEvent : Nat := 151240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩] } }
def leftRaw : List Term := Proof.Events590.exact151234RawTerms
def rightRaw : List Term := Proof.Events589.exact150957RawTerms
def group : MergeGroup := .operator 151234 150957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151234) (leftOrdinal := 0)
    (rightResult := 150957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39234⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151240

namespace LeftMerge151241
def owner : Owner := ⟨.program ⟨257⟩, ⟨39236⟩⟩
def mergeEvent : Nat := 151241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩] } }
def leftRaw : List Term := Proof.Events590.exact151234RawTerms
def rightRaw : List Term := Proof.Events589.exact150957RawTerms
def group : MergeGroup := .operator 151234 150957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151234) (leftOrdinal := 1)
    (rightResult := 150957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39234⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151241

namespace LeftMerge151243
def owner : Owner := ⟨.program ⟨257⟩, ⟨39236⟩⟩
def mergeEvent : Nat := 151243
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38554⟩⟩] } }
def rhsRaw : List Term := Proof.Events589.exact150954RawTerms
def group : MergeGroup := .relation 151242
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151242) (rhsResult := 150954)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39234⟩⟩) ⟨38554⟩ 150954) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38554⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨37404⟩⟩], [⟨.program ⟨257⟩, ⟨38554⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151243

namespace LeftMerge151257
def owner : Owner := ⟨.program ⟨257⟩, ⟨38119⟩⟩
def mergeEvent : Nat := 151257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38116⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events590.exact151251RawTerms
def group : MergeGroup := .operator 149120 151251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 151251) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38116⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38116⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151257

namespace LeftMerge151378
def owner : Owner := ⟨.program ⟨257⟩, ⟨38776⟩⟩
def mergeEvent : Nat := 151378
def frameStart : Nat := 151312
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events591.exact151374RawTerms
def rightRaw : List Term := Proof.Events591.exact151372RawTerms
def group : MergeGroup := .operator 151374 151372
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151374) (leftOrdinal := 0)
    (rightResult := 151372) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37404⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151378

namespace LeftMerge151390
def owner : Owner := ⟨.program ⟨257⟩, ⟨39235⟩⟩
def mergeEvent : Nat := 151390
def frameStart : Nat := 151312
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩] } }
def leftRaw : List Term := Proof.Events591.exact151386RawTerms
def rightRaw : List Term := Proof.Events591.exact151363RawTerms
def group : MergeGroup := .operator 151386 151363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151386) (leftOrdinal := 0)
    (rightResult := 151363) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39234⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151390

namespace LeftMerge151391
def owner : Owner := ⟨.program ⟨257⟩, ⟨39235⟩⟩
def mergeEvent : Nat := 151391
def frameStart : Nat := 151312
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩] } }
def leftRaw : List Term := Proof.Events591.exact151386RawTerms
def rightRaw : List Term := Proof.Events591.exact151363RawTerms
def group : MergeGroup := .operator 151386 151363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151386) (leftOrdinal := 1)
    (rightResult := 151363) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39234⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151391

namespace LeftMerge151393
def owner : Owner := ⟨.program ⟨257⟩, ⟨39235⟩⟩
def mergeEvent : Nat := 151393
def frameStart : Nat := 151312
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37404⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38554⟩⟩] } }
def rhsRaw : List Term := Proof.Events591.exact151360RawTerms
def group : MergeGroup := .relation 151392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151392) (rhsResult := 151360)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37404⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39234⟩⟩) ⟨38554⟩ 151360) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38554⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37404⟩⟩], [⟨.program ⟨257⟩, ⟨38554⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151393

namespace LeftMerge151401
def owner : Owner := ⟨.program ⟨257⟩, ⟨37605⟩⟩
def mergeEvent : Nat := 151401
def frameStart : Nat := 151312
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37604⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events591.exact151374RawTerms
def rightRaw : List Term := Proof.Events591.exact151397RawTerms
def group : MergeGroup := .operator 151374 151397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151374) (leftOrdinal := 0)
    (rightResult := 151397) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37604⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37604⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151401

namespace LeftMerge151418
def owner : Owner := ⟨.program ⟨257⟩, ⟨38119⟩⟩
def mergeEvent : Nat := 151418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }
def rhsRaw : List Term := Proof.Events591.exact151415RawTerms
def group : MergeGroup := .relation 151417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151417) (rhsResult := 151415)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38116⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 151416 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38116⟩⟩]⟩) (none) 151415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151418

namespace LeftMerge151419
def owner : Owner := ⟨.program ⟨257⟩, ⟨38119⟩⟩
def mergeEvent : Nat := 151419
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩] } }
def rhsRaw : List Term := Proof.Events591.exact151415RawTerms
def group : MergeGroup := .relation 151417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151417) (rhsResult := 151415)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38116⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 151416 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38116⟩⟩]⟩) (none) 151415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39234⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151419

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
