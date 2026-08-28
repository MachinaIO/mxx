import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge138342
def owner : Owner := ⟨.program ⟨257⟩, ⟨69164⟩⟩
def mergeEvent : Nat := 138342
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68488⟩⟩] } }
def rhsRaw : List Term := Proof.Events540.exact138267RawTerms
def group : MergeGroup := .relation 138341
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138341) (rhsResult := 138267)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69163⟩⟩) ⟨68488⟩ 138267) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68488⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], [⟨.program ⟨257⟩, ⟨68488⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138342

namespace LeftMerge138343
def owner : Owner := ⟨.program ⟨257⟩, ⟨69164⟩⟩
def mergeEvent : Nat := 138343
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩] } }
def leftRaw : List Term := Proof.Events540.exact138334RawTerms
def rightRaw : List Term := Proof.Events540.exact138270RawTerms
def group : MergeGroup := .operator 138334 138270
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138334) (leftOrdinal := 0)
    (rightResult := 138270) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138343

namespace LeftMerge138357
def owner : Owner := ⟨.program ⟨257⟩, ⟨67703⟩⟩
def mergeEvent : Nat := 138357
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67700⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events540.exact138351RawTerms
def group : MergeGroup := .operator 134495 138351
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 138351) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67700⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67700⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138357

namespace LeftMerge138436
def owner : Owner := ⟨.program ⟨257⟩, ⟨65257⟩⟩
def mergeEvent : Nat := 138436
def frameStart : Nat := 138406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events540.exact138432RawTerms
def rightRaw : List Term := Proof.Events540.exact138429RawTerms
def group : MergeGroup := .operator 138432 138429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138432) (leftOrdinal := 0)
    (rightResult := 138429) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25646⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138436

namespace LeftMerge138466
def owner : Owner := ⟨.program ⟨257⟩, ⟨68901⟩⟩
def mergeEvent : Nat := 138466
def frameStart : Nat := 138406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events540.exact138462RawTerms
def rightRaw : List Term := Proof.Events540.exact138460RawTerms
def group : MergeGroup := .operator 138462 138460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138462) (leftOrdinal := 0)
    (rightResult := 138460) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138466

namespace LeftMerge138489
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def mergeEvent : Nat := 138489
def frameStart : Nat := 138406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events540.exact138485RawTerms
def rightRaw : List Term := Proof.Events540.exact138482RawTerms
def group : MergeGroup := .operator 138485 138482
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138485) (leftOrdinal := 0)
    (rightResult := 138482) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138489

namespace LeftMerge138498
def owner : Owner := ⟨.program ⟨257⟩, ⟨69166⟩⟩
def mergeEvent : Nat := 138498
def frameStart : Nat := 138406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩] } }
def leftRaw : List Term := Proof.Events540.exact138494RawTerms
def rightRaw : List Term := Proof.Events540.exact138451RawTerms
def group : MergeGroup := .operator 138494 138451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138494) (leftOrdinal := 0)
    (rightResult := 138451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69163⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138498

namespace LeftMerge138499
def owner : Owner := ⟨.program ⟨257⟩, ⟨69166⟩⟩
def mergeEvent : Nat := 138499
def frameStart : Nat := 138406
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩] } }
def leftRaw : List Term := Proof.Events540.exact138494RawTerms
def rightRaw : List Term := Proof.Events540.exact138451RawTerms
def group : MergeGroup := .operator 138494 138451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138494) (leftOrdinal := 1)
    (rightResult := 138451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138499

namespace LeftMerge138501
def owner : Owner := ⟨.program ⟨257⟩, ⟨69166⟩⟩
def mergeEvent : Nat := 138501
def frameStart : Nat := 138406
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68488⟩⟩] } }
def rhsRaw : List Term := Proof.Events540.exact138448RawTerms
def group : MergeGroup := .relation 138500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138500) (rhsResult := 138448)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69163⟩⟩) ⟨68488⟩ 138448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68488⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], [⟨.program ⟨257⟩, ⟨68488⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138501

namespace LeftMerge138509
def owner : Owner := ⟨.program ⟨257⟩, ⟨65734⟩⟩
def mergeEvent : Nat := 138509
def frameStart : Nat := 138406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events540.exact138462RawTerms
def rightRaw : List Term := Proof.Events541.exact138505RawTerms
def group : MergeGroup := .operator 138462 138505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138462) (leftOrdinal := 0)
    (rightResult := 138505) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138509

namespace LeftMerge138526
def owner : Owner := ⟨.program ⟨257⟩, ⟨67703⟩⟩
def mergeEvent : Nat := 138526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events541.exact138523RawTerms
def group : MergeGroup := .relation 138525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138525) (rhsResult := 138523)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67700⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 138524 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67700⟩⟩]⟩) (none) 138523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138526

namespace LeftMerge138527
def owner : Owner := ⟨.program ⟨257⟩, ⟨67703⟩⟩
def mergeEvent : Nat := 138527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩] } }
def rhsRaw : List Term := Proof.Events541.exact138523RawTerms
def group : MergeGroup := .relation 138525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138525) (rhsResult := 138523)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67700⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 138524 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67700⟩⟩]⟩) (none) 138523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138527

namespace LeftMerge138528
def owner : Owner := ⟨.program ⟨257⟩, ⟨67703⟩⟩
def mergeEvent : Nat := 138528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68488⟩⟩] } }
def rhsRaw : List Term := Proof.Events541.exact138523RawTerms
def group : MergeGroup := .relation 138525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138525) (rhsResult := 138523)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67700⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 138524 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67700⟩⟩]⟩) (none) 138523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68488⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], [⟨.program ⟨257⟩, ⟨68488⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138528

namespace LeftMerge138529
def owner : Owner := ⟨.program ⟨257⟩, ⟨67703⟩⟩
def mergeEvent : Nat := 138529
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events541.exact138523RawTerms
def group : MergeGroup := .relation 138525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138525) (rhsResult := 138523)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67700⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 138524 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67700⟩⟩]⟩) (none) 138523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138529

namespace LeftMerge138534
def owner : Owner := ⟨.program ⟨257⟩, ⟨69165⟩⟩
def mergeEvent : Nat := 138534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68488⟩⟩] } }
def leftRaw : List Term := Proof.Events541.exact138530RawTerms
def rightRaw : List Term := Proof.Events540.exact138344RawTerms
def group : MergeGroup := .operator 138530 138344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138530) (leftOrdinal := 2)
    (rightResult := 138344) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68488⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68488⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25646⟩⟩, ⟨.program ⟨257⟩, ⟨65256⟩⟩], [⟨.program ⟨257⟩, ⟨68488⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138534

namespace LeftMerge138535
def owner : Owner := ⟨.program ⟨257⟩, ⟨69165⟩⟩
def mergeEvent : Nat := 138535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩] } }
def leftRaw : List Term := Proof.Events541.exact138530RawTerms
def rightRaw : List Term := Proof.Events540.exact138344RawTerms
def group : MergeGroup := .operator 138530 138344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138530) (leftOrdinal := 1)
    (rightResult := 138344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138535

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
