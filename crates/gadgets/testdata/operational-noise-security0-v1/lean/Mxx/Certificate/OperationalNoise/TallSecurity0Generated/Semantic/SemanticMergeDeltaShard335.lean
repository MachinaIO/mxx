import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge55278
def owner : Owner := ⟨.program ⟨214⟩, ⟨19607⟩⟩
def mergeEvent : Nat := 55278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events215.exact55272RawTerms
def group : MergeGroup := .relation 55274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55274) (rhsResult := 55272)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19604⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55273 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19604⟩⟩]⟩) (none) 55272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55278

namespace LeftMerge55283
def owner : Owner := ⟨.program ⟨214⟩, ⟨26150⟩⟩
def mergeEvent : Nat := 55283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23628⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55279RawTerms
def rightRaw : List Term := Proof.Events215.exact55093RawTerms
def group : MergeGroup := .operator 55279 55093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55279) (leftOrdinal := 2)
    (rightResult := 55093) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23628⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23628⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], [⟨.program ⟨214⟩, ⟨23628⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55283

namespace LeftMerge55284
def owner : Owner := ⟨.program ⟨214⟩, ⟨26150⟩⟩
def mergeEvent : Nat := 55284
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55279RawTerms
def rightRaw : List Term := Proof.Events215.exact55093RawTerms
def group : MergeGroup := .operator 55279 55093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55279) (leftOrdinal := 1)
    (rightResult := 55093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26148⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55284

namespace LeftMerge55292
def owner : Owner := ⟨.program ⟨214⟩, ⟨28098⟩⟩
def mergeEvent : Nat := 55292
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55286RawTerms
def rightRaw : List Term := Proof.Events214.exact55009RawTerms
def group : MergeGroup := .operator 55286 55009
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55286) (leftOrdinal := 0)
    (rightResult := 55009) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28096⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55292

namespace LeftMerge55293
def owner : Owner := ⟨.program ⟨214⟩, ⟨28098⟩⟩
def mergeEvent : Nat := 55293
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩] } }
def leftRaw : List Term := Proof.Events215.exact55286RawTerms
def rightRaw : List Term := Proof.Events214.exact55009RawTerms
def group : MergeGroup := .operator 55286 55009
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55286) (leftOrdinal := 1)
    (rightResult := 55009) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28096⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55293

namespace LeftMerge55295
def owner : Owner := ⟨.program ⟨214⟩, ⟨28098⟩⟩
def mergeEvent : Nat := 55295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24228⟩⟩] } }
def rhsRaw : List Term := Proof.Events214.exact55006RawTerms
def group : MergeGroup := .relation 55294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55294) (rhsResult := 55006)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28096⟩⟩) ⟨24228⟩ 55006) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24228⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55295

namespace LeftMerge55309
def owner : Owner := ⟨.program ⟨214⟩, ⟨21551⟩⟩
def mergeEvent : Nat := 55309
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events216.exact55303RawTerms
def group : MergeGroup := .operator 50762 55303
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 55303) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21548⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55309

namespace LeftMerge55430
def owner : Owner := ⟨.program ⟨214⟩, ⟨16140⟩⟩
def mergeEvent : Nat := 55430
def frameStart : Nat := 55364
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55426RawTerms
def rightRaw : List Term := Proof.Events216.exact55424RawTerms
def group : MergeGroup := .operator 55426 55424
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55426) (leftOrdinal := 0)
    (rightResult := 55424) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55430

namespace LeftMerge55442
def owner : Owner := ⟨.program ⟨214⟩, ⟨28097⟩⟩
def mergeEvent : Nat := 55442
def frameStart : Nat := 55364
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55438RawTerms
def rightRaw : List Term := Proof.Events216.exact55415RawTerms
def group : MergeGroup := .operator 55438 55415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55438) (leftOrdinal := 0)
    (rightResult := 55415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28096⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55442

namespace LeftMerge55443
def owner : Owner := ⟨.program ⟨214⟩, ⟨28097⟩⟩
def mergeEvent : Nat := 55443
def frameStart : Nat := 55364
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55438RawTerms
def rightRaw : List Term := Proof.Events216.exact55415RawTerms
def group : MergeGroup := .operator 55438 55415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55438) (leftOrdinal := 1)
    (rightResult := 55415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28096⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55443

namespace LeftMerge55445
def owner : Owner := ⟨.program ⟨214⟩, ⟨28097⟩⟩
def mergeEvent : Nat := 55445
def frameStart : Nat := 55364
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24228⟩⟩] } }
def rhsRaw : List Term := Proof.Events216.exact55412RawTerms
def group : MergeGroup := .relation 55444
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55444) (rhsResult := 55412)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28096⟩⟩) ⟨24228⟩ 55412) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24228⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55445

namespace LeftMerge55453
def owner : Owner := ⟨.program ⟨214⟩, ⟨16109⟩⟩
def mergeEvent : Nat := 55453
def frameStart : Nat := 55364
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16108⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events216.exact55426RawTerms
def rightRaw : List Term := Proof.Events216.exact55449RawTerms
def group : MergeGroup := .operator 55426 55449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 55426) (leftOrdinal := 0)
    (rightResult := 55449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16108⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55453

namespace LeftMerge55470
def owner : Owner := ⟨.program ⟨214⟩, ⟨21551⟩⟩
def mergeEvent : Nat := 55470
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }
def rhsRaw : List Term := Proof.Events216.exact55467RawTerms
def group : MergeGroup := .relation 55469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55469) (rhsResult := 55467)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55468 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩) (none) 55467) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55470

namespace LeftMerge55471
def owner : Owner := ⟨.program ⟨214⟩, ⟨21551⟩⟩
def mergeEvent : Nat := 55471
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩] } }
def rhsRaw : List Term := Proof.Events216.exact55467RawTerms
def group : MergeGroup := .relation 55469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55469) (rhsResult := 55467)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55468 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩) (none) 55467) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55471

namespace LeftMerge55472
def owner : Owner := ⟨.program ⟨214⟩, ⟨21551⟩⟩
def mergeEvent : Nat := 55472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24228⟩⟩] } }
def rhsRaw : List Term := Proof.Events216.exact55467RawTerms
def group : MergeGroup := .relation 55469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55469) (rhsResult := 55467)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55468 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩) (none) 55467) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16063⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24228⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge55472

namespace LeftMerge55473
def owner : Owner := ⟨.program ⟨214⟩, ⟨21551⟩⟩
def mergeEvent : Nat := 55473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16108⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events216.exact55467RawTerms
def group : MergeGroup := .relation 55469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 55469) (rhsResult := 55467)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 55468 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩) (none) 55467) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16108⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge55473

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
