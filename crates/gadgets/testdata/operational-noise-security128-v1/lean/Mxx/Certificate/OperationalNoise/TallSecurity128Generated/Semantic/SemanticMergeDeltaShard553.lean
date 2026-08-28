import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge92224
def owner : Owner := ⟨.program ⟨257⟩, ⟨40150⟩⟩
def mergeEvent : Nat := 92224
def frameStart : Nat := 92121
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events360.exact92177RawTerms
def rightRaw : List Term := Proof.Events360.exact92220RawTerms
def group : MergeGroup := .operator 92177 92220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92177) (leftOrdinal := 0)
    (rightResult := 92220) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92224

namespace LeftMerge92241
def owner : Owner := ⟨.program ⟨257⟩, ⟨40602⟩⟩
def mergeEvent : Nat := 92241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events360.exact92238RawTerms
def group : MergeGroup := .relation 92240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92240) (rhsResult := 92238)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40599⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 92239 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40599⟩⟩]⟩) (none) 92238) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92241

namespace LeftMerge92242
def owner : Owner := ⟨.program ⟨257⟩, ⟨40602⟩⟩
def mergeEvent : Nat := 92242
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩] } }
def rhsRaw : List Term := Proof.Events360.exact92238RawTerms
def group : MergeGroup := .relation 92240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92240) (rhsResult := 92238)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40599⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 92239 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40599⟩⟩]⟩) (none) 92238) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92242

namespace LeftMerge92243
def owner : Owner := ⟨.program ⟨257⟩, ⟨40602⟩⟩
def mergeEvent : Nat := 92243
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41139⟩⟩] } }
def rhsRaw : List Term := Proof.Events360.exact92238RawTerms
def group : MergeGroup := .relation 92240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92240) (rhsResult := 92238)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40599⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 92239 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40599⟩⟩]⟩) (none) 92238) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41139⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92243

namespace LeftMerge92244
def owner : Owner := ⟨.program ⟨257⟩, ⟨40602⟩⟩
def mergeEvent : Nat := 92244
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events360.exact92238RawTerms
def group : MergeGroup := .relation 92240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92240) (rhsResult := 92238)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40599⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 92239 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40599⟩⟩]⟩) (none) 92238) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92244

namespace LeftMerge92249
def owner : Owner := ⟨.program ⟨257⟩, ⟨41676⟩⟩
def mergeEvent : Nat := 92249
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41139⟩⟩] } }
def leftRaw : List Term := Proof.Events360.exact92245RawTerms
def rightRaw : List Term := Proof.Events359.exact92059RawTerms
def group : MergeGroup := .operator 92245 92059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92245) (leftOrdinal := 2)
    (rightResult := 92059) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41139⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14256⟩⟩, ⟨.program ⟨257⟩, ⟨39914⟩⟩], [⟨.program ⟨257⟩, ⟨41139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92249

namespace LeftMerge92250
def owner : Owner := ⟨.program ⟨257⟩, ⟨41676⟩⟩
def mergeEvent : Nat := 92250
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩] } }
def leftRaw : List Term := Proof.Events360.exact92245RawTerms
def rightRaw : List Term := Proof.Events359.exact92059RawTerms
def group : MergeGroup := .operator 92245 92059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92245) (leftOrdinal := 1)
    (rightResult := 92059) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92250

namespace LeftMerge92258
def owner : Owner := ⟨.program ⟨257⟩, ⟨42116⟩⟩
def mergeEvent : Nat := 92258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩] } }
def leftRaw : List Term := Proof.Events360.exact92252RawTerms
def rightRaw : List Term := Proof.Events359.exact91975RawTerms
def group : MergeGroup := .operator 92252 91975
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92252) (leftOrdinal := 0)
    (rightResult := 91975) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42114⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92258

namespace LeftMerge92259
def owner : Owner := ⟨.program ⟨257⟩, ⟨42116⟩⟩
def mergeEvent : Nat := 92259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩] } }
def leftRaw : List Term := Proof.Events360.exact92252RawTerms
def rightRaw : List Term := Proof.Events359.exact91975RawTerms
def group : MergeGroup := .operator 92252 91975
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92252) (leftOrdinal := 1)
    (rightResult := 91975) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42114⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92259

namespace LeftMerge92261
def owner : Owner := ⟨.program ⟨257⟩, ⟨42116⟩⟩
def mergeEvent : Nat := 92261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41306⟩⟩] } }
def rhsRaw : List Term := Proof.Events359.exact91972RawTerms
def group : MergeGroup := .relation 92260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92260) (rhsResult := 91972)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42114⟩⟩) ⟨41306⟩ 91972) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41306⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨41306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92261

namespace LeftMerge92275
def owner : Owner := ⟨.program ⟨257⟩, ⟨40959⟩⟩
def mergeEvent : Nat := 92275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40956⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events360.exact92269RawTerms
def group : MergeGroup := .operator 90620 92269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 92269) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40956⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40956⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92275

namespace LeftMerge92396
def owner : Owner := ⟨.program ⟨257⟩, ⟨41488⟩⟩
def mergeEvent : Nat := 92396
def frameStart : Nat := 92330
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events360.exact92392RawTerms
def rightRaw : List Term := Proof.Events360.exact92390RawTerms
def group : MergeGroup := .operator 92392 92390
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92392) (leftOrdinal := 0)
    (rightResult := 92390) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92396

namespace LeftMerge92408
def owner : Owner := ⟨.program ⟨257⟩, ⟨42115⟩⟩
def mergeEvent : Nat := 92408
def frameStart : Nat := 92330
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩] } }
def leftRaw : List Term := Proof.Events360.exact92404RawTerms
def rightRaw : List Term := Proof.Events360.exact92381RawTerms
def group : MergeGroup := .operator 92404 92381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92404) (leftOrdinal := 0)
    (rightResult := 92381) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42114⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92408

namespace LeftMerge92409
def owner : Owner := ⟨.program ⟨257⟩, ⟨42115⟩⟩
def mergeEvent : Nat := 92409
def frameStart : Nat := 92330
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩] } }
def leftRaw : List Term := Proof.Events360.exact92404RawTerms
def rightRaw : List Term := Proof.Events360.exact92381RawTerms
def group : MergeGroup := .operator 92404 92381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92404) (leftOrdinal := 1)
    (rightResult := 92381) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42114⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92409

namespace LeftMerge92411
def owner : Owner := ⟨.program ⟨257⟩, ⟨42115⟩⟩
def mergeEvent : Nat := 92411
def frameStart : Nat := 92330
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41306⟩⟩] } }
def rhsRaw : List Term := Proof.Events360.exact92378RawTerms
def group : MergeGroup := .relation 92410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92410) (rhsResult := 92378)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42114⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42114⟩⟩) ⟨41306⟩ 92378) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41306⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40148⟩⟩], [⟨.program ⟨257⟩, ⟨41306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92411

namespace LeftMerge92419
def owner : Owner := ⟨.program ⟨257⟩, ⟨40385⟩⟩
def mergeEvent : Nat := 92419
def frameStart : Nat := 92330
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40384⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events360.exact92392RawTerms
def rightRaw : List Term := Proof.Events360.exact92415RawTerms
def group : MergeGroup := .operator 92392 92415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92392) (leftOrdinal := 0)
    (rightResult := 92415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40384⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40384⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92419

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
