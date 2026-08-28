import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge65028
def owner : Owner := ⟨.program ⟨214⟩, ⟨30150⟩⟩
def mergeEvent : Nat := 65028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩] } }
def leftRaw : List Term := Proof.Events253.exact64974RawTerms
def rightRaw : List Term := Proof.Events023.exact6081RawTerms
def group : MergeGroup := .operator 64974 6081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 64974) (leftOrdinal := 11)
    (rightResult := 6081) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7825⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65028

namespace LeftMerge65030
def owner : Owner := ⟨.program ⟨214⟩, ⟨30150⟩⟩
def mergeEvent : Nat := 65030
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def rhsRaw : List Term := Proof.Events023.exact6074RawTerms
def group : MergeGroup := .relation 65029
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65029) (rhsResult := 6074)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7825⟩⟩) ⟨6752⟩ 6074) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15213⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65030

namespace LeftMerge65031
def owner : Owner := ⟨.program ⟨214⟩, ⟨30150⟩⟩
def mergeEvent : Nat := 65031
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩] } }
def leftRaw : List Term := Proof.Events253.exact64974RawTerms
def rightRaw : List Term := Proof.Events023.exact6081RawTerms
def group : MergeGroup := .operator 64974 6081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 64974) (leftOrdinal := 15)
    (rightResult := 6081) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7825⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65031

namespace LeftMerge65033
def owner : Owner := ⟨.program ⟨214⟩, ⟨30150⟩⟩
def mergeEvent : Nat := 65033
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def rhsRaw : List Term := Proof.Events023.exact6074RawTerms
def group : MergeGroup := .relation 65032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65032) (rhsResult := 6074)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7825⟩⟩) ⟨6752⟩ 6074) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15052⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65033

namespace LeftMerge65034
def owner : Owner := ⟨.program ⟨214⟩, ⟨30150⟩⟩
def mergeEvent : Nat := 65034
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩] } }
def leftRaw : List Term := Proof.Events253.exact64974RawTerms
def rightRaw : List Term := Proof.Events023.exact6081RawTerms
def group : MergeGroup := .operator 64974 6081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 64974) (leftOrdinal := 18)
    (rightResult := 6081) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7825⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65034

namespace LeftMerge65036
def owner : Owner := ⟨.program ⟨214⟩, ⟨30150⟩⟩
def mergeEvent : Nat := 65036
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def rhsRaw : List Term := Proof.Events023.exact6074RawTerms
def group : MergeGroup := .relation 65035
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65035) (rhsResult := 6074)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7825⟩⟩) ⟨6752⟩ 6074) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14891⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65036

namespace LeftMerge65037
def owner : Owner := ⟨.program ⟨214⟩, ⟨30150⟩⟩
def mergeEvent : Nat := 65037
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩] } }
def leftRaw : List Term := Proof.Events253.exact64974RawTerms
def rightRaw : List Term := Proof.Events023.exact6081RawTerms
def group : MergeGroup := .operator 64974 6081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 64974) (leftOrdinal := 0)
    (rightResult := 6081) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7825⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65037

namespace LeftMerge65042
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def mergeEvent : Nat := 65042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65038RawTerms
def rightRaw : List Term := Proof.Events197.exact50635RawTerms
def group : MergeGroup := .operator 65038 50635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65038) (leftOrdinal := 6)
    (rightResult := 50635) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65042

namespace LeftMerge65043
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def mergeEvent : Nat := 65043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65038RawTerms
def rightRaw : List Term := Proof.Events197.exact50635RawTerms
def group : MergeGroup := .operator 65038 50635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65038) (leftOrdinal := 8)
    (rightResult := 50635) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65043

namespace LeftMerge65044
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def mergeEvent : Nat := 65044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65038RawTerms
def rightRaw : List Term := Proof.Events197.exact50635RawTerms
def group : MergeGroup := .operator 65038 50635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65038) (leftOrdinal := 9)
    (rightResult := 50635) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65044

namespace LeftMerge65045
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def mergeEvent : Nat := 65045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65038RawTerms
def rightRaw : List Term := Proof.Events197.exact50635RawTerms
def group : MergeGroup := .operator 65038 50635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65038) (leftOrdinal := 10)
    (rightResult := 50635) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65045

namespace LeftMerge65046
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def mergeEvent : Nat := 65046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65038RawTerms
def rightRaw : List Term := Proof.Events197.exact50635RawTerms
def group : MergeGroup := .operator 65038 50635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65038) (leftOrdinal := 12)
    (rightResult := 50635) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65046

namespace LeftMerge65047
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def mergeEvent : Nat := 65047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65038RawTerms
def rightRaw : List Term := Proof.Events197.exact50635RawTerms
def group : MergeGroup := .operator 65038 50635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65038) (leftOrdinal := 13)
    (rightResult := 50635) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65047

namespace LeftMerge65048
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def mergeEvent : Nat := 65048
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65038RawTerms
def rightRaw : List Term := Proof.Events197.exact50635RawTerms
def group : MergeGroup := .operator 65038 50635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65038) (leftOrdinal := 14)
    (rightResult := 50635) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65048

namespace LeftMerge65049
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def mergeEvent : Nat := 65049
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65038RawTerms
def rightRaw : List Term := Proof.Events197.exact50635RawTerms
def group : MergeGroup := .operator 65038 50635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65038) (leftOrdinal := 16)
    (rightResult := 50635) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65049

namespace LeftMerge65050
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def mergeEvent : Nat := 65050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65038RawTerms
def rightRaw : List Term := Proof.Events197.exact50635RawTerms
def group : MergeGroup := .operator 65038 50635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65038) (leftOrdinal := 17)
    (rightResult := 50635) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6752⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65050

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
