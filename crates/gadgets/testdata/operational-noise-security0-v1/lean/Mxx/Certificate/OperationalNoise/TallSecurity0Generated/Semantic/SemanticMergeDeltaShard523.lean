import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge85281
def owner : Owner := ⟨.program ⟨214⟩, ⟨25990⟩⟩
def mergeEvent : Nat := 85281
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85275RawTerms
def rightRaw : List Term := Proof.Events332.exact85211RawTerms
def group : MergeGroup := .operator 85275 85211
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85275) (leftOrdinal := 1)
    (rightResult := 85211) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25989⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85281

namespace LeftMerge85283
def owner : Owner := ⟨.program ⟨214⟩, ⟨25990⟩⟩
def mergeEvent : Nat := 85283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23542⟩⟩] } }
def rhsRaw : List Term := Proof.Events332.exact85208RawTerms
def group : MergeGroup := .relation 85282
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85282) (rhsResult := 85208)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25989⟩⟩) ⟨23542⟩ 85208) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23542⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨23542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85283

namespace LeftMerge85284
def owner : Owner := ⟨.program ⟨214⟩, ⟨25990⟩⟩
def mergeEvent : Nat := 85284
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85275RawTerms
def rightRaw : List Term := Proof.Events332.exact85211RawTerms
def group : MergeGroup := .operator 85275 85211
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85275) (leftOrdinal := 0)
    (rightResult := 85211) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25989⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85284

namespace LeftMerge85298
def owner : Owner := ⟨.program ⟨214⟩, ⟨19459⟩⟩
def mergeEvent : Nat := 85298
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19456⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events333.exact85292RawTerms
def group : MergeGroup := .operator 80012 85292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 85292) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19456⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19456⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85298

namespace LeftMerge85377
def owner : Owner := ⟨.program ⟨214⟩, ⟨13991⟩⟩
def mergeEvent : Nat := 85377
def frameStart : Nat := 85347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events333.exact85373RawTerms
def rightRaw : List Term := Proof.Events333.exact85370RawTerms
def group : MergeGroup := .operator 85373 85370
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85373) (leftOrdinal := 0)
    (rightResult := 85370) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11385⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85377

namespace LeftMerge85407
def owner : Owner := ⟨.program ⟨214⟩, ⟨14099⟩⟩
def mergeEvent : Nat := 85407
def frameStart : Nat := 85347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85403RawTerms
def rightRaw : List Term := Proof.Events333.exact85401RawTerms
def group : MergeGroup := .operator 85403 85401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85403) (leftOrdinal := 0)
    (rightResult := 85401) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85407

namespace LeftMerge85428
def owner : Owner := ⟨.program ⟨214⟩, ⟨7851⟩⟩
def mergeEvent : Nat := 85428
def frameStart : Nat := 85347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85424RawTerms
def rightRaw : List Term := Proof.Events333.exact85421RawTerms
def group : MergeGroup := .operator 85424 85421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85424) (leftOrdinal := 0)
    (rightResult := 85421) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7849⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85428

namespace LeftMerge85437
def owner : Owner := ⟨.program ⟨214⟩, ⟨25992⟩⟩
def mergeEvent : Nat := 85437
def frameStart : Nat := 85347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85433RawTerms
def rightRaw : List Term := Proof.Events333.exact85392RawTerms
def group : MergeGroup := .operator 85433 85392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85433) (leftOrdinal := 0)
    (rightResult := 85392) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25989⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85437

namespace LeftMerge85438
def owner : Owner := ⟨.program ⟨214⟩, ⟨25992⟩⟩
def mergeEvent : Nat := 85438
def frameStart : Nat := 85347
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85433RawTerms
def rightRaw : List Term := Proof.Events333.exact85392RawTerms
def group : MergeGroup := .operator 85433 85392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85433) (leftOrdinal := 1)
    (rightResult := 85392) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25989⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85438

namespace LeftMerge85440
def owner : Owner := ⟨.program ⟨214⟩, ⟨25992⟩⟩
def mergeEvent : Nat := 85440
def frameStart : Nat := 85347
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23542⟩⟩] } }
def rhsRaw : List Term := Proof.Events333.exact85389RawTerms
def group : MergeGroup := .relation 85439
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85439) (rhsResult := 85389)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25989⟩⟩) ⟨23542⟩ 85389) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23542⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨23542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85440

namespace LeftMerge85448
def owner : Owner := ⟨.program ⟨214⟩, ⟨15823⟩⟩
def mergeEvent : Nat := 85448
def frameStart : Nat := 85347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85403RawTerms
def rightRaw : List Term := Proof.Events333.exact85444RawTerms
def group : MergeGroup := .operator 85403 85444
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85403) (leftOrdinal := 0)
    (rightResult := 85444) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85448

namespace LeftMerge85465
def owner : Owner := ⟨.program ⟨214⟩, ⟨19459⟩⟩
def mergeEvent : Nat := 85465
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }
def rhsRaw : List Term := Proof.Events333.exact85462RawTerms
def group : MergeGroup := .relation 85464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85464) (rhsResult := 85462)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19456⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85463 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19456⟩⟩]⟩) (none) 85462) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85465

namespace LeftMerge85466
def owner : Owner := ⟨.program ⟨214⟩, ⟨19459⟩⟩
def mergeEvent : Nat := 85466
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩] } }
def rhsRaw : List Term := Proof.Events333.exact85462RawTerms
def group : MergeGroup := .relation 85464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85464) (rhsResult := 85462)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19456⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85463 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19456⟩⟩]⟩) (none) 85462) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85466

namespace LeftMerge85467
def owner : Owner := ⟨.program ⟨214⟩, ⟨19459⟩⟩
def mergeEvent : Nat := 85467
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23542⟩⟩] } }
def rhsRaw : List Term := Proof.Events333.exact85462RawTerms
def group : MergeGroup := .relation 85464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85464) (rhsResult := 85462)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19456⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85463 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19456⟩⟩]⟩) (none) 85462) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23542⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨23542⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85467

namespace LeftMerge85468
def owner : Owner := ⟨.program ⟨214⟩, ⟨19459⟩⟩
def mergeEvent : Nat := 85468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events333.exact85462RawTerms
def group : MergeGroup := .relation 85464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85464) (rhsResult := 85462)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19456⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85463 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19456⟩⟩]⟩) (none) 85462) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85468

namespace LeftMerge85473
def owner : Owner := ⟨.program ⟨214⟩, ⟨25991⟩⟩
def mergeEvent : Nat := 85473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23542⟩⟩] } }
def leftRaw : List Term := Proof.Events333.exact85469RawTerms
def rightRaw : List Term := Proof.Events333.exact85285RawTerms
def group : MergeGroup := .operator 85469 85285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85469) (leftOrdinal := 2)
    (rightResult := 85285) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23542⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23542⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], [⟨.program ⟨214⟩, ⟨23542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85473

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
