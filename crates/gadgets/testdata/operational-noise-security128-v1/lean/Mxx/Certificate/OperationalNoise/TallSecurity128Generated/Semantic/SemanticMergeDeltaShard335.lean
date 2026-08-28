import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge58210
def owner : Owner := ⟨.program ⟨257⟩, ⟨36826⟩⟩
def mergeEvent : Nat := 58210
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨34812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35972⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58205RawTerms
def rightRaw : List Term := Proof.Events226.exact58027RawTerms
def group : MergeGroup := .operator 58205 58027
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58205) (leftOrdinal := 2)
    (rightResult := 58027) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨34812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35972⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨34812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35972⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨34812⟩⟩], [⟨.program ⟨257⟩, ⟨35972⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58210

namespace LeftMerge58218
def owner : Owner := ⟨.program ⟨257⟩, ⟨36827⟩⟩
def mergeEvent : Nat := 58218
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58212RawTerms
def rightRaw : List Term := Proof.Events061.exact15642RawTerms
def group : MergeGroup := .operator 58212 15642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58212) (leftOrdinal := 0)
    (rightResult := 15642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58218

namespace LeftMerge58219
def owner : Owner := ⟨.program ⟨257⟩, ⟨36827⟩⟩
def mergeEvent : Nat := 58219
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩] } }
def leftRaw : List Term := Proof.Events227.exact58212RawTerms
def rightRaw : List Term := Proof.Events061.exact15642RawTerms
def group : MergeGroup := .operator 58212 15642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58212) (leftOrdinal := 1)
    (rightResult := 15642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58219

namespace LeftMerge58221
def owner : Owner := ⟨.program ⟨257⟩, ⟨36827⟩⟩
def mergeEvent : Nat := 58221
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15635RawTerms
def group : MergeGroup := .relation 58220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58220) (rhsResult := 15635)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7163⟩⟩) ⟨7047⟩ 15635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58221

namespace LeftMerge58235
def owner : Owner := ⟨.program ⟨257⟩, ⟨31165⟩⟩
def mergeEvent : Nat := 58235
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49823RawTerms
def rightRaw : List Term := Proof.Events227.exact58229RawTerms
def group : MergeGroup := .operator 49823 58229
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49823) (leftOrdinal := 0)
    (rightResult := 58229) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58235

namespace LeftMerge58236
def owner : Owner := ⟨.program ⟨257⟩, ⟨31165⟩⟩
def mergeEvent : Nat := 58236
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49823RawTerms
def rightRaw : List Term := Proof.Events227.exact58229RawTerms
def group : MergeGroup := .operator 49823 58229
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49823) (leftOrdinal := 1)
    (rightResult := 58229) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58236

namespace LeftMerge58238
def owner : Owner := ⟨.program ⟨257⟩, ⟨31165⟩⟩
def mergeEvent : Nat := 58238
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30312⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58226RawTerms
def group : MergeGroup := .relation 58237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58237) (rhsResult := 58226)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31163⟩⟩) ⟨30312⟩ 58226) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30312⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58238

namespace LeftMerge58252
def owner : Owner := ⟨.program ⟨257⟩, ⟨29995⟩⟩
def mergeEvent : Nat := 58252
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29992⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events227.exact58246RawTerms
def group : MergeGroup := .operator 46745 58246
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 58246) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29992⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29992⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58252

namespace LeftMerge58373
def owner : Owner := ⟨.program ⟨257⟩, ⟨30480⟩⟩
def mergeEvent : Nat := 58373
def frameStart : Nat := 58307
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events228.exact58369RawTerms
def rightRaw : List Term := Proof.Events227.exact58367RawTerms
def group : MergeGroup := .operator 58369 58367
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58369) (leftOrdinal := 0)
    (rightResult := 58367) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58373

namespace LeftMerge58385
def owner : Owner := ⟨.program ⟨257⟩, ⟨31164⟩⟩
def mergeEvent : Nat := 58385
def frameStart : Nat := 58307
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩] } }
def leftRaw : List Term := Proof.Events228.exact58381RawTerms
def rightRaw : List Term := Proof.Events227.exact58358RawTerms
def group : MergeGroup := .operator 58381 58358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58381) (leftOrdinal := 0)
    (rightResult := 58358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31163⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58385

namespace LeftMerge58386
def owner : Owner := ⟨.program ⟨257⟩, ⟨31164⟩⟩
def mergeEvent : Nat := 58386
def frameStart : Nat := 58307
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩] } }
def leftRaw : List Term := Proof.Events228.exact58381RawTerms
def rightRaw : List Term := Proof.Events227.exact58358RawTerms
def group : MergeGroup := .operator 58381 58358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58381) (leftOrdinal := 1)
    (rightResult := 58358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58386

namespace LeftMerge58388
def owner : Owner := ⟨.program ⟨257⟩, ⟨31164⟩⟩
def mergeEvent : Nat := 58388
def frameStart : Nat := 58307
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30312⟩⟩] } }
def rhsRaw : List Term := Proof.Events227.exact58355RawTerms
def group : MergeGroup := .relation 58387
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58387) (rhsResult := 58355)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31163⟩⟩) ⟨30312⟩ 58355) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30312⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58388

namespace LeftMerge58396
def owner : Owner := ⟨.program ⟨257⟩, ⟨29408⟩⟩
def mergeEvent : Nat := 58396
def frameStart : Nat := 58307
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29406⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events228.exact58369RawTerms
def rightRaw : List Term := Proof.Events228.exact58392RawTerms
def group : MergeGroup := .operator 58369 58392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58369) (leftOrdinal := 0)
    (rightResult := 58392) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29406⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58396

namespace LeftMerge58413
def owner : Owner := ⟨.program ⟨257⟩, ⟨29995⟩⟩
def mergeEvent : Nat := 58413
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }
def rhsRaw : List Term := Proof.Events228.exact58410RawTerms
def group : MergeGroup := .relation 58412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58412) (rhsResult := 58410)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29992⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58411 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29992⟩⟩]⟩) (none) 58410) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58413

namespace LeftMerge58414
def owner : Owner := ⟨.program ⟨257⟩, ⟨29995⟩⟩
def mergeEvent : Nat := 58414
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩] } }
def rhsRaw : List Term := Proof.Events228.exact58410RawTerms
def group : MergeGroup := .relation 58412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58412) (rhsResult := 58410)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29992⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58411 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29992⟩⟩]⟩) (none) 58410) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58414

namespace LeftMerge58415
def owner : Owner := ⟨.program ⟨257⟩, ⟨29995⟩⟩
def mergeEvent : Nat := 58415
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30312⟩⟩] } }
def rhsRaw : List Term := Proof.Events228.exact58410RawTerms
def group : MergeGroup := .relation 58412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58412) (rhsResult := 58410)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29992⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58411 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29992⟩⟩]⟩) (none) 58410) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30312⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58415

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
