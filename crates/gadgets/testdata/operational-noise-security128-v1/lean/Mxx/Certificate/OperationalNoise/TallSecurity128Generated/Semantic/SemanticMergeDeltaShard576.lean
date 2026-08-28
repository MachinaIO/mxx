import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge96393
def owner : Owner := ⟨.program ⟨257⟩, ⟨55555⟩⟩
def mergeEvent : Nat := 96393
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩] } }
def leftRaw : List Term := Proof.Events376.exact96387RawTerms
def rightRaw : List Term := Proof.Events376.exact96323RawTerms
def group : MergeGroup := .operator 96387 96323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96387) (leftOrdinal := 1)
    (rightResult := 96323) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55554⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96393

namespace LeftMerge96395
def owner : Owner := ⟨.program ⟨257⟩, ⟨55555⟩⟩
def mergeEvent : Nat := 96395
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55019⟩⟩] } }
def rhsRaw : List Term := Proof.Events376.exact96320RawTerms
def group : MergeGroup := .relation 96394
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96394) (rhsResult := 96320)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55554⟩⟩) ⟨55019⟩ 96320) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55019⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], [⟨.program ⟨257⟩, ⟨55019⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96395

namespace LeftMerge96396
def owner : Owner := ⟨.program ⟨257⟩, ⟨55555⟩⟩
def mergeEvent : Nat := 96396
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩] } }
def leftRaw : List Term := Proof.Events376.exact96387RawTerms
def rightRaw : List Term := Proof.Events376.exact96323RawTerms
def group : MergeGroup := .operator 96387 96323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96387) (leftOrdinal := 0)
    (rightResult := 96323) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55554⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96396

namespace LeftMerge96410
def owner : Owner := ⟨.program ⟨257⟩, ⟨54482⟩⟩
def mergeEvent : Nat := 96410
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54479⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events376.exact96404RawTerms
def group : MergeGroup := .operator 90620 96404
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 96404) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54479⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54479⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96410

namespace LeftMerge96489
def owner : Owner := ⟨.program ⟨257⟩, ⟨53661⟩⟩
def mergeEvent : Nat := 96489
def frameStart : Nat := 96459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events376.exact96485RawTerms
def rightRaw : List Term := Proof.Events376.exact96482RawTerms
def group : MergeGroup := .operator 96485 96482
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96485) (leftOrdinal := 0)
    (rightResult := 96482) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24830⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96489

namespace LeftMerge96519
def owner : Owner := ⟨.program ⟨257⟩, ⟨55288⟩⟩
def mergeEvent : Nat := 96519
def frameStart : Nat := 96459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events377.exact96515RawTerms
def rightRaw : List Term := Proof.Events377.exact96513RawTerms
def group : MergeGroup := .operator 96515 96513
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96515) (leftOrdinal := 0)
    (rightResult := 96513) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96519

namespace LeftMerge96542
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 96542
def frameStart : Nat := 96459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events377.exact96538RawTerms
def rightRaw : List Term := Proof.Events377.exact96535RawTerms
def group : MergeGroup := .operator 96538 96535
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96538) (leftOrdinal := 0)
    (rightResult := 96535) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96542

namespace LeftMerge96551
def owner : Owner := ⟨.program ⟨257⟩, ⟨55557⟩⟩
def mergeEvent : Nat := 96551
def frameStart : Nat := 96459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩] } }
def leftRaw : List Term := Proof.Events377.exact96547RawTerms
def rightRaw : List Term := Proof.Events376.exact96504RawTerms
def group : MergeGroup := .operator 96547 96504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96547) (leftOrdinal := 0)
    (rightResult := 96504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55554⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96551

namespace LeftMerge96552
def owner : Owner := ⟨.program ⟨257⟩, ⟨55557⟩⟩
def mergeEvent : Nat := 96552
def frameStart : Nat := 96459
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩] } }
def leftRaw : List Term := Proof.Events377.exact96547RawTerms
def rightRaw : List Term := Proof.Events376.exact96504RawTerms
def group : MergeGroup := .operator 96547 96504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96547) (leftOrdinal := 1)
    (rightResult := 96504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55554⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96552

namespace LeftMerge96554
def owner : Owner := ⟨.program ⟨257⟩, ⟨55557⟩⟩
def mergeEvent : Nat := 96554
def frameStart : Nat := 96459
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55019⟩⟩] } }
def rhsRaw : List Term := Proof.Events376.exact96501RawTerms
def group : MergeGroup := .relation 96553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96553) (rhsResult := 96501)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55554⟩⟩) ⟨55019⟩ 96501) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55019⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], [⟨.program ⟨257⟩, ⟨55019⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96554

namespace LeftMerge96562
def owner : Owner := ⟨.program ⟨257⟩, ⟨53910⟩⟩
def mergeEvent : Nat := 96562
def frameStart : Nat := 96459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events377.exact96515RawTerms
def rightRaw : List Term := Proof.Events377.exact96558RawTerms
def group : MergeGroup := .operator 96515 96558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96515) (leftOrdinal := 0)
    (rightResult := 96558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53908⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96562

namespace LeftMerge96579
def owner : Owner := ⟨.program ⟨257⟩, ⟨54482⟩⟩
def mergeEvent : Nat := 96579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events377.exact96576RawTerms
def group : MergeGroup := .relation 96578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96578) (rhsResult := 96576)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96577 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54479⟩⟩]⟩) (none) 96576) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96579

namespace LeftMerge96580
def owner : Owner := ⟨.program ⟨257⟩, ⟨54482⟩⟩
def mergeEvent : Nat := 96580
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩] } }
def rhsRaw : List Term := Proof.Events377.exact96576RawTerms
def group : MergeGroup := .relation 96578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96578) (rhsResult := 96576)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96577 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54479⟩⟩]⟩) (none) 96576) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55554⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96580

namespace LeftMerge96581
def owner : Owner := ⟨.program ⟨257⟩, ⟨54482⟩⟩
def mergeEvent : Nat := 96581
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55019⟩⟩] } }
def rhsRaw : List Term := Proof.Events377.exact96576RawTerms
def group : MergeGroup := .relation 96578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96578) (rhsResult := 96576)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96577 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54479⟩⟩]⟩) (none) 96576) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55019⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], [⟨.program ⟨257⟩, ⟨55019⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96581

namespace LeftMerge96582
def owner : Owner := ⟨.program ⟨257⟩, ⟨54482⟩⟩
def mergeEvent : Nat := 96582
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨53908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events377.exact96576RawTerms
def group : MergeGroup := .relation 96578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96578) (rhsResult := 96576)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96577 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54479⟩⟩]⟩) (none) 96576) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨53908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96582

namespace LeftMerge96587
def owner : Owner := ⟨.program ⟨257⟩, ⟨55556⟩⟩
def mergeEvent : Nat := 96587
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55019⟩⟩] } }
def leftRaw : List Term := Proof.Events377.exact96583RawTerms
def rightRaw : List Term := Proof.Events376.exact96397RawTerms
def group : MergeGroup := .operator 96583 96397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96583) (leftOrdinal := 2)
    (rightResult := 96397) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55019⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55019⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24830⟩⟩, ⟨.program ⟨257⟩, ⟨53660⟩⟩], [⟨.program ⟨257⟩, ⟨55019⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96587

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
