import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge43964
def owner : Owner := ⟨.program ⟨214⟩, ⟨10782⟩⟩
def mergeEvent : Nat := 43964
def frameStart : Nat := 43904
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events171.exact43960RawTerms
def rightRaw : List Term := Proof.Events171.exact43958RawTerms
def group : MergeGroup := .operator 43960 43958
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43960) (leftOrdinal := 0)
    (rightResult := 43958) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43964

namespace LeftMerge43987
def owner : Owner := ⟨.program ⟨214⟩, ⟨7836⟩⟩
def mergeEvent : Nat := 43987
def frameStart : Nat := 43904
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }
def leftRaw : List Term := Proof.Events171.exact43983RawTerms
def rightRaw : List Term := Proof.Events171.exact43980RawTerms
def group : MergeGroup := .operator 43983 43980
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43983) (leftOrdinal := 0)
    (rightResult := 43980) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7834⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43987

namespace LeftMerge43996
def owner : Owner := ⟨.program ⟨214⟩, ⟨25001⟩⟩
def mergeEvent : Nat := 43996
def frameStart : Nat := 43904
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩] } }
def leftRaw : List Term := Proof.Events171.exact43992RawTerms
def rightRaw : List Term := Proof.Events171.exact43949RawTerms
def group : MergeGroup := .operator 43992 43949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43992) (leftOrdinal := 0)
    (rightResult := 43949) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24998⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43996

namespace LeftMerge43997
def owner : Owner := ⟨.program ⟨214⟩, ⟨25001⟩⟩
def mergeEvent : Nat := 43997
def frameStart : Nat := 43904
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩] } }
def leftRaw : List Term := Proof.Events171.exact43992RawTerms
def rightRaw : List Term := Proof.Events171.exact43949RawTerms
def group : MergeGroup := .operator 43992 43949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43992) (leftOrdinal := 1)
    (rightResult := 43949) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24998⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43997

namespace LeftMerge43999
def owner : Owner := ⟨.program ⟨214⟩, ⟨25001⟩⟩
def mergeEvent : Nat := 43999
def frameStart : Nat := 43904
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23000⟩⟩] } }
def rhsRaw : List Term := Proof.Events171.exact43946RawTerms
def group : MergeGroup := .relation 43998
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43998) (rhsResult := 43946)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24998⟩⟩) ⟨23000⟩ 43946) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23000⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], [⟨.program ⟨214⟩, ⟨23000⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43999

namespace LeftMerge44007
def owner : Owner := ⟨.program ⟨214⟩, ⟨14963⟩⟩
def mergeEvent : Nat := 44007
def frameStart : Nat := 43904
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14961⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events171.exact43960RawTerms
def rightRaw : List Term := Proof.Events171.exact44003RawTerms
def group : MergeGroup := .operator 43960 44003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43960) (leftOrdinal := 0)
    (rightResult := 44003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14961⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44007

namespace LeftMerge44024
def owner : Owner := ⟨.program ⟨214⟩, ⟨19107⟩⟩
def mergeEvent : Nat := 44024
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }
def rhsRaw : List Term := Proof.Events171.exact44021RawTerms
def group : MergeGroup := .relation 44023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44023) (rhsResult := 44021)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19104⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 44022 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19104⟩⟩]⟩) (none) 44021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44024

namespace LeftMerge44025
def owner : Owner := ⟨.program ⟨214⟩, ⟨19107⟩⟩
def mergeEvent : Nat := 44025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩] } }
def rhsRaw : List Term := Proof.Events171.exact44021RawTerms
def group : MergeGroup := .relation 44023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44023) (rhsResult := 44021)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19104⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 44022 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19104⟩⟩]⟩) (none) 44021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44025

namespace LeftMerge44026
def owner : Owner := ⟨.program ⟨214⟩, ⟨19107⟩⟩
def mergeEvent : Nat := 44026
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23000⟩⟩] } }
def rhsRaw : List Term := Proof.Events171.exact44021RawTerms
def group : MergeGroup := .relation 44023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44023) (rhsResult := 44021)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19104⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 44022 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19104⟩⟩]⟩) (none) 44021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23000⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], [⟨.program ⟨214⟩, ⟨23000⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44026

namespace LeftMerge44027
def owner : Owner := ⟨.program ⟨214⟩, ⟨19107⟩⟩
def mergeEvent : Nat := 44027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events171.exact44021RawTerms
def group : MergeGroup := .relation 44023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44023) (rhsResult := 44021)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19104⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 44022 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19104⟩⟩]⟩) (none) 44021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14961⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44027

namespace LeftMerge44032
def owner : Owner := ⟨.program ⟨214⟩, ⟨25000⟩⟩
def mergeEvent : Nat := 44032
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23000⟩⟩] } }
def leftRaw : List Term := Proof.Events171.exact44028RawTerms
def rightRaw : List Term := Proof.Events171.exact43842RawTerms
def group : MergeGroup := .operator 44028 43842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44028) (leftOrdinal := 2)
    (rightResult := 43842) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23000⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23000⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], [⟨.program ⟨214⟩, ⟨23000⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44032

namespace LeftMerge44033
def owner : Owner := ⟨.program ⟨214⟩, ⟨25000⟩⟩
def mergeEvent : Nat := 44033
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩] } }
def leftRaw : List Term := Proof.Events171.exact44028RawTerms
def rightRaw : List Term := Proof.Events171.exact43842RawTerms
def group : MergeGroup := .operator 44028 43842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44028) (leftOrdinal := 1)
    (rightResult := 43842) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44033

namespace LeftMerge44041
def owner : Owner := ⟨.program ⟨214⟩, ⟨26592⟩⟩
def mergeEvent : Nat := 44041
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩] } }
def leftRaw : List Term := Proof.Events172.exact44035RawTerms
def rightRaw : List Term := Proof.Events170.exact43758RawTerms
def group : MergeGroup := .operator 44035 43758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44035) (leftOrdinal := 0)
    (rightResult := 43758) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26590⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44041

namespace LeftMerge44042
def owner : Owner := ⟨.program ⟨214⟩, ⟨26592⟩⟩
def mergeEvent : Nat := 44042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩] } }
def leftRaw : List Term := Proof.Events172.exact44035RawTerms
def rightRaw : List Term := Proof.Events170.exact43758RawTerms
def group : MergeGroup := .operator 44035 43758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44035) (leftOrdinal := 1)
    (rightResult := 43758) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26590⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44042

namespace LeftMerge44044
def owner : Owner := ⟨.program ⟨214⟩, ⟨26592⟩⟩
def mergeEvent : Nat := 44044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23790⟩⟩] } }
def rhsRaw : List Term := Proof.Events170.exact43755RawTerms
def group : MergeGroup := .relation 44043
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44043) (rhsResult := 43755)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26590⟩⟩) ⟨23790⟩ 43755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23790⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44044

namespace LeftMerge44058
def owner : Owner := ⟨.program ⟨214⟩, ⟨20547⟩⟩
def mergeEvent : Nat := 44058
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20544⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events172.exact44052RawTerms
def group : MergeGroup := .operator 36137 44052
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 44052) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44058

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
