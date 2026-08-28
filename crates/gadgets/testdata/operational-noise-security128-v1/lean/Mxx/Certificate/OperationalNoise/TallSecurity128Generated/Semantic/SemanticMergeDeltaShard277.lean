import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge48182
def owner : Owner := ⟨.program ⟨257⟩, ⟨41708⟩⟩
def mergeEvent : Nat := 48182
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41157⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact48107RawTerms
def group : MergeGroup := .relation 48181
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48181) (rhsResult := 48107)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41707⟩⟩) ⟨41157⟩ 48107) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41157⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨41157⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48182

namespace LeftMerge48183
def owner : Owner := ⟨.program ⟨257⟩, ⟨41708⟩⟩
def mergeEvent : Nat := 48183
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩] } }
def leftRaw : List Term := Proof.Events188.exact48174RawTerms
def rightRaw : List Term := Proof.Events187.exact48110RawTerms
def group : MergeGroup := .operator 48174 48110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48174) (leftOrdinal := 0)
    (rightResult := 48110) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41707⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48183

namespace LeftMerge48197
def owner : Owner := ⟨.program ⟨257⟩, ⟨40632⟩⟩
def mergeEvent : Nat := 48197
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40629⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events188.exact48191RawTerms
def group : MergeGroup := .operator 46745 48191
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 48191) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40629⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40629⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48197

namespace LeftMerge48276
def owner : Owner := ⟨.program ⟨257⟩, ⟨39987⟩⟩
def mergeEvent : Nat := 48276
def frameStart : Nat := 48246
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events188.exact48272RawTerms
def rightRaw : List Term := Proof.Events188.exact48269RawTerms
def group : MergeGroup := .operator 48272 48269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48272) (leftOrdinal := 0)
    (rightResult := 48269) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14301⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48276

namespace LeftMerge48306
def owner : Owner := ⟨.program ⟨257⟩, ⟨41420⟩⟩
def mergeEvent : Nat := 48306
def frameStart : Nat := 48246
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events188.exact48302RawTerms
def rightRaw : List Term := Proof.Events188.exact48300RawTerms
def group : MergeGroup := .operator 48302 48300
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48302) (leftOrdinal := 0)
    (rightResult := 48300) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48306

namespace LeftMerge48329
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 48329
def frameStart : Nat := 48246
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events188.exact48325RawTerms
def rightRaw : List Term := Proof.Events188.exact48322RawTerms
def group : MergeGroup := .operator 48325 48322
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48325) (leftOrdinal := 0)
    (rightResult := 48322) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48329

namespace LeftMerge48338
def owner : Owner := ⟨.program ⟨257⟩, ⟨41710⟩⟩
def mergeEvent : Nat := 48338
def frameStart : Nat := 48246
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩] } }
def leftRaw : List Term := Proof.Events188.exact48334RawTerms
def rightRaw : List Term := Proof.Events188.exact48291RawTerms
def group : MergeGroup := .operator 48334 48291
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48334) (leftOrdinal := 0)
    (rightResult := 48291) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41707⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48338

namespace LeftMerge48339
def owner : Owner := ⟨.program ⟨257⟩, ⟨41710⟩⟩
def mergeEvent : Nat := 48339
def frameStart : Nat := 48246
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩] } }
def leftRaw : List Term := Proof.Events188.exact48334RawTerms
def rightRaw : List Term := Proof.Events188.exact48291RawTerms
def group : MergeGroup := .operator 48334 48291
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48334) (leftOrdinal := 1)
    (rightResult := 48291) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41707⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48339

namespace LeftMerge48341
def owner : Owner := ⟨.program ⟨257⟩, ⟨41710⟩⟩
def mergeEvent : Nat := 48341
def frameStart : Nat := 48246
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41157⟩⟩] } }
def rhsRaw : List Term := Proof.Events188.exact48288RawTerms
def group : MergeGroup := .relation 48340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48340) (rhsResult := 48288)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41707⟩⟩) ⟨41157⟩ 48288) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41157⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨41157⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48341

namespace LeftMerge48349
def owner : Owner := ⟨.program ⟨257⟩, ⟨40174⟩⟩
def mergeEvent : Nat := 48349
def frameStart : Nat := 48246
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events188.exact48302RawTerms
def rightRaw : List Term := Proof.Events188.exact48345RawTerms
def group : MergeGroup := .operator 48302 48345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48302) (leftOrdinal := 0)
    (rightResult := 48345) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40172⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48349

namespace LeftMerge48366
def owner : Owner := ⟨.program ⟨257⟩, ⟨40632⟩⟩
def mergeEvent : Nat := 48366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events188.exact48363RawTerms
def group : MergeGroup := .relation 48365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48365) (rhsResult := 48363)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40629⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48364 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40629⟩⟩]⟩) (none) 48363) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48366

namespace LeftMerge48367
def owner : Owner := ⟨.program ⟨257⟩, ⟨40632⟩⟩
def mergeEvent : Nat := 48367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩] } }
def rhsRaw : List Term := Proof.Events188.exact48363RawTerms
def group : MergeGroup := .relation 48365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48365) (rhsResult := 48363)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40629⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48364 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40629⟩⟩]⟩) (none) 48363) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48367

namespace LeftMerge48368
def owner : Owner := ⟨.program ⟨257⟩, ⟨40632⟩⟩
def mergeEvent : Nat := 48368
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41157⟩⟩] } }
def rhsRaw : List Term := Proof.Events188.exact48363RawTerms
def group : MergeGroup := .relation 48365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48365) (rhsResult := 48363)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40629⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48364 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40629⟩⟩]⟩) (none) 48363) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41157⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨41157⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48368

namespace LeftMerge48369
def owner : Owner := ⟨.program ⟨257⟩, ⟨40632⟩⟩
def mergeEvent : Nat := 48369
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events188.exact48363RawTerms
def group : MergeGroup := .relation 48365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48365) (rhsResult := 48363)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40629⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48364 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40629⟩⟩]⟩) (none) 48363) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48369

namespace LeftMerge48374
def owner : Owner := ⟨.program ⟨257⟩, ⟨41709⟩⟩
def mergeEvent : Nat := 48374
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41157⟩⟩] } }
def leftRaw : List Term := Proof.Events188.exact48370RawTerms
def rightRaw : List Term := Proof.Events188.exact48184RawTerms
def group : MergeGroup := .operator 48370 48184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48370) (leftOrdinal := 2)
    (rightResult := 48184) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41157⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41157⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨41157⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48374

namespace LeftMerge48375
def owner : Owner := ⟨.program ⟨257⟩, ⟨41709⟩⟩
def mergeEvent : Nat := 48375
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩] } }
def leftRaw : List Term := Proof.Events188.exact48370RawTerms
def rightRaw : List Term := Proof.Events188.exact48184RawTerms
def group : MergeGroup := .operator 48370 48184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48370) (leftOrdinal := 1)
    (rightResult := 48184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41707⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48375

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
