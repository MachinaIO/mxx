import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge252160
def owner : Owner := ⟨.program ⟨257⟩, ⟨46926⟩⟩
def mergeEvent : Nat := 252160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46439⟩⟩] } }
def leftRaw : List Term := Proof.Events984.exact252156RawTerms
def rightRaw : List Term := Proof.Events984.exact251970RawTerms
def group : MergeGroup := .operator 252156 251970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252156) (leftOrdinal := 2)
    (rightResult := 251970) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46439⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46439⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], [⟨.program ⟨257⟩, ⟨46439⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252160

namespace LeftMerge252161
def owner : Owner := ⟨.program ⟨257⟩, ⟨46926⟩⟩
def mergeEvent : Nat := 252161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩] } }
def leftRaw : List Term := Proof.Events984.exact252156RawTerms
def rightRaw : List Term := Proof.Events984.exact251970RawTerms
def group : MergeGroup := .operator 252156 251970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252156) (leftOrdinal := 1)
    (rightResult := 251970) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252161

namespace LeftMerge252169
def owner : Owner := ⟨.program ⟨257⟩, ⟨47226⟩⟩
def mergeEvent : Nat := 252169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩] } }
def leftRaw : List Term := Proof.Events985.exact252163RawTerms
def rightRaw : List Term := Proof.Events983.exact251886RawTerms
def group : MergeGroup := .operator 252163 251886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252163) (leftOrdinal := 0)
    (rightResult := 251886) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47224⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252169

namespace LeftMerge252170
def owner : Owner := ⟨.program ⟨257⟩, ⟨47226⟩⟩
def mergeEvent : Nat := 252170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩] } }
def leftRaw : List Term := Proof.Events985.exact252163RawTerms
def rightRaw : List Term := Proof.Events983.exact251886RawTerms
def group : MergeGroup := .operator 252163 251886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252163) (leftOrdinal := 1)
    (rightResult := 251886) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47224⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252170

namespace LeftMerge252172
def owner : Owner := ⟨.program ⟨257⟩, ⟨47226⟩⟩
def mergeEvent : Nat := 252172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46576⟩⟩] } }
def rhsRaw : List Term := Proof.Events983.exact251883RawTerms
def group : MergeGroup := .relation 252171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252171) (rhsResult := 251883)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47224⟩⟩) ⟨46576⟩ 251883) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46576⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252172

namespace LeftMerge252186
def owner : Owner := ⟨.program ⟨257⟩, ⟨46119⟩⟩
def mergeEvent : Nat := 252186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events985.exact252180RawTerms
def group : MergeGroup := .operator 251495 252180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 252180) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46116⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252186

namespace LeftMerge252307
def owner : Owner := ⟨.program ⟨257⟩, ⟨46808⟩⟩
def mergeEvent : Nat := 252307
def frameStart : Nat := 252241
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events985.exact252303RawTerms
def rightRaw : List Term := Proof.Events985.exact252301RawTerms
def group : MergeGroup := .operator 252303 252301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252303) (leftOrdinal := 0)
    (rightResult := 252301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252307

namespace LeftMerge252319
def owner : Owner := ⟨.program ⟨257⟩, ⟨47225⟩⟩
def mergeEvent : Nat := 252319
def frameStart : Nat := 252241
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩] } }
def leftRaw : List Term := Proof.Events985.exact252315RawTerms
def rightRaw : List Term := Proof.Events985.exact252292RawTerms
def group : MergeGroup := .operator 252315 252292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252315) (leftOrdinal := 0)
    (rightResult := 252292) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47224⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252319

namespace LeftMerge252320
def owner : Owner := ⟨.program ⟨257⟩, ⟨47225⟩⟩
def mergeEvent : Nat := 252320
def frameStart : Nat := 252241
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩] } }
def leftRaw : List Term := Proof.Events985.exact252315RawTerms
def rightRaw : List Term := Proof.Events985.exact252292RawTerms
def group : MergeGroup := .operator 252315 252292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252315) (leftOrdinal := 1)
    (rightResult := 252292) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47224⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252320

namespace LeftMerge252322
def owner : Owner := ⟨.program ⟨257⟩, ⟨47225⟩⟩
def mergeEvent : Nat := 252322
def frameStart : Nat := 252241
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46576⟩⟩] } }
def rhsRaw : List Term := Proof.Events985.exact252289RawTerms
def group : MergeGroup := .relation 252321
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252321) (rhsResult := 252289)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47224⟩⟩) ⟨46576⟩ 252289) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46576⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252322

namespace LeftMerge252330
def owner : Owner := ⟨.program ⟨257⟩, ⟨45619⟩⟩
def mergeEvent : Nat := 252330
def frameStart : Nat := 252241
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events985.exact252303RawTerms
def rightRaw : List Term := Proof.Events985.exact252326RawTerms
def group : MergeGroup := .operator 252303 252326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252303) (leftOrdinal := 0)
    (rightResult := 252326) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252330

namespace LeftMerge252347
def owner : Owner := ⟨.program ⟨257⟩, ⟨46119⟩⟩
def mergeEvent : Nat := 252347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }
def rhsRaw : List Term := Proof.Events985.exact252344RawTerms
def group : MergeGroup := .relation 252346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252346) (rhsResult := 252344)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 252345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩) (none) 252344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252347

namespace LeftMerge252348
def owner : Owner := ⟨.program ⟨257⟩, ⟨46119⟩⟩
def mergeEvent : Nat := 252348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩] } }
def rhsRaw : List Term := Proof.Events985.exact252344RawTerms
def group : MergeGroup := .relation 252346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252346) (rhsResult := 252344)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 252345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩) (none) 252344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252348

namespace LeftMerge252349
def owner : Owner := ⟨.program ⟨257⟩, ⟨46119⟩⟩
def mergeEvent : Nat := 252349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46576⟩⟩] } }
def rhsRaw : List Term := Proof.Events985.exact252344RawTerms
def group : MergeGroup := .relation 252346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252346) (rhsResult := 252344)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 252345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩) (none) 252344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46576⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252349

namespace LeftMerge252350
def owner : Owner := ⟨.program ⟨257⟩, ⟨46119⟩⟩
def mergeEvent : Nat := 252350
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events985.exact252344RawTerms
def group : MergeGroup := .relation 252346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 252346) (rhsResult := 252344)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 252345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩) (none) 252344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge252350

namespace LeftMerge252355
def owner : Owner := ⟨.program ⟨257⟩, ⟨47227⟩⟩
def mergeEvent : Nat := 252355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩] } }
def leftRaw : List Term := Proof.Events985.exact252351RawTerms
def rightRaw : List Term := Proof.Events985.exact252173RawTerms
def group : MergeGroup := .operator 252351 252173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 252351) (leftOrdinal := 0)
    (rightResult := 252173) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge252355

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
