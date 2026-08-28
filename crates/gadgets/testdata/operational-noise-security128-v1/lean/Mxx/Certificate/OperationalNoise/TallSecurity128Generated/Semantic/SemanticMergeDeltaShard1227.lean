import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge200372
def owner : Owner := ⟨.program ⟨257⟩, ⟨23464⟩⟩
def mergeEvent : Nat := 200372
def frameStart : Nat := 200280
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩] } }
def leftRaw : List Term := Proof.Events782.exact200368RawTerms
def rightRaw : List Term := Proof.Events782.exact200325RawTerms
def group : MergeGroup := .operator 200368 200325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 200368) (leftOrdinal := 0)
    (rightResult := 200325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23461⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge200372

namespace LeftMerge200373
def owner : Owner := ⟨.program ⟨257⟩, ⟨23464⟩⟩
def mergeEvent : Nat := 200373
def frameStart : Nat := 200280
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩] } }
def leftRaw : List Term := Proof.Events782.exact200368RawTerms
def rightRaw : List Term := Proof.Events782.exact200325RawTerms
def group : MergeGroup := .operator 200368 200325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 200368) (leftOrdinal := 1)
    (rightResult := 200325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23461⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge200373

namespace LeftMerge200375
def owner : Owner := ⟨.program ⟨257⟩, ⟨23464⟩⟩
def mergeEvent : Nat := 200375
def frameStart : Nat := 200280
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22941⟩⟩] } }
def rhsRaw : List Term := Proof.Events782.exact200322RawTerms
def group : MergeGroup := .relation 200374
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 200374) (rhsResult := 200322)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23461⟩⟩) ⟨22941⟩ 200322) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22941⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], [⟨.program ⟨257⟩, ⟨22941⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge200375

namespace LeftMerge200383
def owner : Owner := ⟨.program ⟨257⟩, ⟨21826⟩⟩
def mergeEvent : Nat := 200383
def frameStart : Nat := 200280
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events782.exact200336RawTerms
def rightRaw : List Term := Proof.Events782.exact200379RawTerms
def group : MergeGroup := .operator 200336 200379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 200336) (leftOrdinal := 0)
    (rightResult := 200379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge200383

namespace LeftMerge200400
def owner : Owner := ⟨.program ⟨257⟩, ⟨22392⟩⟩
def mergeEvent : Nat := 200400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events782.exact200397RawTerms
def group : MergeGroup := .relation 200399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 200399) (rhsResult := 200397)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 200398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22389⟩⟩]⟩) (none) 200397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge200400

namespace LeftMerge200401
def owner : Owner := ⟨.program ⟨257⟩, ⟨22392⟩⟩
def mergeEvent : Nat := 200401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩] } }
def rhsRaw : List Term := Proof.Events782.exact200397RawTerms
def group : MergeGroup := .relation 200399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 200399) (rhsResult := 200397)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 200398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22389⟩⟩]⟩) (none) 200397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge200401

namespace LeftMerge200402
def owner : Owner := ⟨.program ⟨257⟩, ⟨22392⟩⟩
def mergeEvent : Nat := 200402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22941⟩⟩] } }
def rhsRaw : List Term := Proof.Events782.exact200397RawTerms
def group : MergeGroup := .relation 200399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 200399) (rhsResult := 200397)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 200398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22389⟩⟩]⟩) (none) 200397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22941⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], [⟨.program ⟨257⟩, ⟨22941⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge200402

namespace LeftMerge200403
def owner : Owner := ⟨.program ⟨257⟩, ⟨22392⟩⟩
def mergeEvent : Nat := 200403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events782.exact200397RawTerms
def group : MergeGroup := .relation 200399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 200399) (rhsResult := 200397)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 200398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22389⟩⟩]⟩) (none) 200397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge200403

namespace LeftMerge200408
def owner : Owner := ⟨.program ⟨257⟩, ⟨23463⟩⟩
def mergeEvent : Nat := 200408
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22941⟩⟩] } }
def leftRaw : List Term := Proof.Events782.exact200404RawTerms
def rightRaw : List Term := Proof.Events782.exact200218RawTerms
def group : MergeGroup := .operator 200404 200218
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 200404) (leftOrdinal := 2)
    (rightResult := 200218) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22941⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22941⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21131⟩⟩, ⟨.program ⟨257⟩, ⟨21542⟩⟩], [⟨.program ⟨257⟩, ⟨22941⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge200408

namespace LeftMerge200409
def owner : Owner := ⟨.program ⟨257⟩, ⟨23463⟩⟩
def mergeEvent : Nat := 200409
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩] } }
def leftRaw : List Term := Proof.Events782.exact200404RawTerms
def rightRaw : List Term := Proof.Events782.exact200218RawTerms
def group : MergeGroup := .operator 200404 200218
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 200404) (leftOrdinal := 1)
    (rightResult := 200218) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23461⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge200409

namespace LeftMerge200417
def owner : Owner := ⟨.program ⟨257⟩, ⟨23936⟩⟩
def mergeEvent : Nat := 200417
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23934⟩⟩] } }
def leftRaw : List Term := Proof.Events782.exact200411RawTerms
def rightRaw : List Term := Proof.Events781.exact200134RawTerms
def group : MergeGroup := .operator 200411 200134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 200411) (leftOrdinal := 0)
    (rightResult := 200134) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge200417

namespace LeftMerge200418
def owner : Owner := ⟨.program ⟨257⟩, ⟨23936⟩⟩
def mergeEvent : Nat := 200418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23934⟩⟩] } }
def leftRaw : List Term := Proof.Events782.exact200411RawTerms
def rightRaw : List Term := Proof.Events781.exact200134RawTerms
def group : MergeGroup := .operator 200411 200134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 200411) (leftOrdinal := 1)
    (rightResult := 200134) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge200418

namespace LeftMerge200420
def owner : Owner := ⟨.program ⟨257⟩, ⟨23936⟩⟩
def mergeEvent : Nat := 200420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23099⟩⟩] } }
def rhsRaw : List Term := Proof.Events781.exact200131RawTerms
def group : MergeGroup := .relation 200419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 200419) (rhsResult := 200131)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23934⟩⟩) ⟨23099⟩ 200131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23099⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨21824⟩⟩], [⟨.program ⟨257⟩, ⟨23099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge200420

namespace LeftMerge200434
def owner : Owner := ⟨.program ⟨257⟩, ⟨22719⟩⟩
def mergeEvent : Nat := 200434
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22716⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events782.exact200428RawTerms
def group : MergeGroup := .operator 192995 200428
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 200428) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22716⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22716⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge200434

namespace LeftMerge200555
def owner : Owner := ⟨.program ⟨257⟩, ⟨23296⟩⟩
def mergeEvent : Nat := 200555
def frameStart : Nat := 200489
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events783.exact200551RawTerms
def rightRaw : List Term := Proof.Events783.exact200549RawTerms
def group : MergeGroup := .operator 200551 200549
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 200551) (leftOrdinal := 0)
    (rightResult := 200549) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge200555

namespace LeftMerge200567
def owner : Owner := ⟨.program ⟨257⟩, ⟨23935⟩⟩
def mergeEvent : Nat := 200567
def frameStart : Nat := 200489
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23934⟩⟩] } }
def leftRaw : List Term := Proof.Events783.exact200563RawTerms
def rightRaw : List Term := Proof.Events783.exact200540RawTerms
def group : MergeGroup := .operator 200563 200540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 200563) (leftOrdinal := 0)
    (rightResult := 200540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge200567

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
