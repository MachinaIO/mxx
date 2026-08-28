import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge165341
def owner : Owner := ⟨.program ⟨257⟩, ⟨41666⟩⟩
def mergeEvent : Nat := 165341
def frameStart : Nat := 165246
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41133⟩⟩] } }
def rhsRaw : List Term := Proof.Events645.exact165288RawTerms
def group : MergeGroup := .relation 165340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165340) (rhsResult := 165288)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41663⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41663⟩⟩) ⟨41133⟩ 165288) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41133⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], [⟨.program ⟨257⟩, ⟨41133⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165341

namespace LeftMerge165349
def owner : Owner := ⟨.program ⟨257⟩, ⟨40142⟩⟩
def mergeEvent : Nat := 165349
def frameStart : Nat := 165246
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events645.exact165302RawTerms
def rightRaw : List Term := Proof.Events645.exact165345RawTerms
def group : MergeGroup := .operator 165302 165345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165302) (leftOrdinal := 0)
    (rightResult := 165345) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165349

namespace LeftMerge165366
def owner : Owner := ⟨.program ⟨257⟩, ⟨40592⟩⟩
def mergeEvent : Nat := 165366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events645.exact165363RawTerms
def group : MergeGroup := .relation 165365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165365) (rhsResult := 165363)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 165364 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40589⟩⟩]⟩) (none) 165363) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165366

namespace LeftMerge165367
def owner : Owner := ⟨.program ⟨257⟩, ⟨40592⟩⟩
def mergeEvent : Nat := 165367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41663⟩⟩] } }
def rhsRaw : List Term := Proof.Events645.exact165363RawTerms
def group : MergeGroup := .relation 165365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165365) (rhsResult := 165363)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 165364 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40589⟩⟩]⟩) (none) 165363) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41663⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41663⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165367

namespace LeftMerge165368
def owner : Owner := ⟨.program ⟨257⟩, ⟨40592⟩⟩
def mergeEvent : Nat := 165368
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41133⟩⟩] } }
def rhsRaw : List Term := Proof.Events645.exact165363RawTerms
def group : MergeGroup := .relation 165365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165365) (rhsResult := 165363)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 165364 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40589⟩⟩]⟩) (none) 165363) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41133⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], [⟨.program ⟨257⟩, ⟨41133⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165368

namespace LeftMerge165369
def owner : Owner := ⟨.program ⟨257⟩, ⟨40592⟩⟩
def mergeEvent : Nat := 165369
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events645.exact165363RawTerms
def group : MergeGroup := .relation 165365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165365) (rhsResult := 165363)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 165364 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40589⟩⟩]⟩) (none) 165363) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165369

namespace LeftMerge165374
def owner : Owner := ⟨.program ⟨257⟩, ⟨41665⟩⟩
def mergeEvent : Nat := 165374
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41133⟩⟩] } }
def leftRaw : List Term := Proof.Events645.exact165370RawTerms
def rightRaw : List Term := Proof.Events645.exact165184RawTerms
def group : MergeGroup := .operator 165370 165184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165370) (leftOrdinal := 2)
    (rightResult := 165184) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41133⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41133⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], [⟨.program ⟨257⟩, ⟨41133⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165374

namespace LeftMerge165375
def owner : Owner := ⟨.program ⟨257⟩, ⟨41665⟩⟩
def mergeEvent : Nat := 165375
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41663⟩⟩] } }
def leftRaw : List Term := Proof.Events645.exact165370RawTerms
def rightRaw : List Term := Proof.Events645.exact165184RawTerms
def group : MergeGroup := .operator 165370 165184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165370) (leftOrdinal := 1)
    (rightResult := 165184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41663⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41663⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41663⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165375

namespace LeftMerge165383
def owner : Owner := ⟨.program ⟨257⟩, ⟨42091⟩⟩
def mergeEvent : Nat := 165383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42089⟩⟩] } }
def leftRaw : List Term := Proof.Events646.exact165377RawTerms
def rightRaw : List Term := Proof.Events644.exact165100RawTerms
def group : MergeGroup := .operator 165377 165100
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165377) (leftOrdinal := 0)
    (rightResult := 165100) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42089⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42089⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165383

namespace LeftMerge165384
def owner : Owner := ⟨.program ⟨257⟩, ⟨42091⟩⟩
def mergeEvent : Nat := 165384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42089⟩⟩] } }
def leftRaw : List Term := Proof.Events646.exact165377RawTerms
def rightRaw : List Term := Proof.Events644.exact165100RawTerms
def group : MergeGroup := .operator 165377 165100
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165377) (leftOrdinal := 1)
    (rightResult := 165100) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42089⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42089⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165384

namespace LeftMerge165386
def owner : Owner := ⟨.program ⟨257⟩, ⟨42091⟩⟩
def mergeEvent : Nat := 165386
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41297⟩⟩] } }
def rhsRaw : List Term := Proof.Events644.exact165097RawTerms
def group : MergeGroup := .relation 165385
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165385) (rhsResult := 165097)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42089⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42089⟩⟩) ⟨41297⟩ 165097) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41297⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40140⟩⟩], [⟨.program ⟨257⟩, ⟨41297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165386

namespace LeftMerge165400
def owner : Owner := ⟨.program ⟨257⟩, ⟨40939⟩⟩
def mergeEvent : Nat := 165400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40936⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events646.exact165394RawTerms
def group : MergeGroup := .operator 163745 165394
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 165394) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40936⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40936⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165400

namespace LeftMerge165521
def owner : Owner := ⟨.program ⟨257⟩, ⟨41484⟩⟩
def mergeEvent : Nat := 165521
def frameStart : Nat := 165455
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events646.exact165517RawTerms
def rightRaw : List Term := Proof.Events646.exact165515RawTerms
def group : MergeGroup := .operator 165517 165515
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165517) (leftOrdinal := 0)
    (rightResult := 165515) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165521

namespace LeftMerge165533
def owner : Owner := ⟨.program ⟨257⟩, ⟨42090⟩⟩
def mergeEvent : Nat := 165533
def frameStart : Nat := 165455
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42089⟩⟩] } }
def leftRaw : List Term := Proof.Events646.exact165529RawTerms
def rightRaw : List Term := Proof.Events646.exact165506RawTerms
def group : MergeGroup := .operator 165529 165506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165529) (leftOrdinal := 0)
    (rightResult := 165506) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42089⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42089⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165533

namespace LeftMerge165534
def owner : Owner := ⟨.program ⟨257⟩, ⟨42090⟩⟩
def mergeEvent : Nat := 165534
def frameStart : Nat := 165455
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42089⟩⟩] } }
def leftRaw : List Term := Proof.Events646.exact165529RawTerms
def rightRaw : List Term := Proof.Events646.exact165506RawTerms
def group : MergeGroup := .operator 165529 165506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165529) (leftOrdinal := 1)
    (rightResult := 165506) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42089⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42089⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165534

namespace LeftMerge165536
def owner : Owner := ⟨.program ⟨257⟩, ⟨42090⟩⟩
def mergeEvent : Nat := 165536
def frameStart : Nat := 165455
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40140⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41297⟩⟩] } }
def rhsRaw : List Term := Proof.Events646.exact165503RawTerms
def group : MergeGroup := .relation 165535
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165535) (rhsResult := 165503)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40140⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42089⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42089⟩⟩) ⟨41297⟩ 165503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41297⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40140⟩⟩], [⟨.program ⟨257⟩, ⟨41297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165536

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
