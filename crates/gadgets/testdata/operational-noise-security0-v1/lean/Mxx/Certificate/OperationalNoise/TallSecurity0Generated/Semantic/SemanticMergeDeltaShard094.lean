import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge17073
def owner : Owner := ⟨.program ⟨214⟩, ⟨30211⟩⟩
def mergeEvent : Nat := 17073
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact17038RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 17038 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17038) (leftOrdinal := 2)
    (rightResult := 15622) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17073

namespace LeftMerge17074
def owner : Owner := ⟨.program ⟨214⟩, ⟨30211⟩⟩
def mergeEvent : Nat := 17074
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact17038RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 17038 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17038) (leftOrdinal := 20)
    (rightResult := 15622) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17074

namespace LeftMerge17075
def owner : Owner := ⟨.program ⟨214⟩, ⟨30211⟩⟩
def mergeEvent : Nat := 17075
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact17038RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 17038 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17038) (leftOrdinal := 1)
    (rightResult := 15622) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17075

namespace LeftMerge17076
def owner : Owner := ⟨.program ⟨214⟩, ⟨30211⟩⟩
def mergeEvent : Nat := 17076
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact17038RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 17038 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17038) (leftOrdinal := 19)
    (rightResult := 15622) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17076

namespace LeftMerge17077
def owner : Owner := ⟨.program ⟨214⟩, ⟨30211⟩⟩
def mergeEvent : Nat := 17077
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact17038RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 17038 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17038) (leftOrdinal := 0)
    (rightResult := 15622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17077

namespace LeftMerge17085
def owner : Owner := ⟨.program ⟨214⟩, ⟨30212⟩⟩
def mergeEvent : Nat := 17085
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact17079RawTerms
def rightRaw : List Term := Proof.Events021.exact5499RawTerms
def group : MergeGroup := .operator 17079 5499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17079) (leftOrdinal := 0)
    (rightResult := 5499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6651⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17085

namespace LeftMerge17086
def owner : Owner := ⟨.program ⟨214⟩, ⟨30212⟩⟩
def mergeEvent : Nat := 17086
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact17079RawTerms
def rightRaw : List Term := Proof.Events021.exact5499RawTerms
def group : MergeGroup := .operator 17079 5499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17079) (leftOrdinal := 1)
    (rightResult := 5499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6651⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17086

namespace LeftMerge17088
def owner : Owner := ⟨.program ⟨214⟩, ⟨30212⟩⟩
def mergeEvent : Nat := 17088
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events021.exact5492RawTerms
def group : MergeGroup := .relation 17087
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17087) (rhsResult := 5492)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6651⟩⟩) ⟨6597⟩ 5492) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17088

namespace LeftMerge17102
def owner : Owner := ⟨.program ⟨214⟩, ⟨30200⟩⟩
def mergeEvent : Nat := 17102
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6747RawTerms
def rightRaw : List Term := Proof.Events066.exact17096RawTerms
def group : MergeGroup := .operator 6747 17096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6747) (leftOrdinal := 1)
    (rightResult := 17096) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30198⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17102

namespace LeftMerge17104
def owner : Owner := ⟨.program ⟨214⟩, ⟨30200⟩⟩
def mergeEvent : Nat := 17104
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24803⟩⟩] } }
def rhsRaw : List Term := Proof.Events066.exact17093RawTerms
def group : MergeGroup := .relation 17103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17103) (rhsResult := 17093)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30198⟩⟩) ⟨24803⟩ 17093) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24803⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨24803⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17104

namespace LeftMerge17105
def owner : Owner := ⟨.program ⟨214⟩, ⟨30200⟩⟩
def mergeEvent : Nat := 17105
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6747RawTerms
def rightRaw : List Term := Proof.Events066.exact17096RawTerms
def group : MergeGroup := .operator 6747 17096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6747) (leftOrdinal := 0)
    (rightResult := 17096) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30198⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17105

namespace LeftMerge17119
def owner : Owner := ⟨.program ⟨214⟩, ⟨22787⟩⟩
def mergeEvent : Nat := 17119
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22784⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events066.exact17113RawTerms
def group : MergeGroup := .operator 6561 17113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 17113) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22784⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22784⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17119

namespace LeftMerge17240
def owner : Owner := ⟨.program ⟨214⟩, ⟨17069⟩⟩
def mergeEvent : Nat := 17240
def frameStart : Nat := 17174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17236RawTerms
def rightRaw : List Term := Proof.Events067.exact17234RawTerms
def group : MergeGroup := .operator 17236 17234
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17236) (leftOrdinal := 0)
    (rightResult := 17234) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17240

namespace LeftMerge17252
def owner : Owner := ⟨.program ⟨214⟩, ⟨30199⟩⟩
def mergeEvent : Nat := 17252
def frameStart : Nat := 17174
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17248RawTerms
def rightRaw : List Term := Proof.Events067.exact17225RawTerms
def group : MergeGroup := .operator 17248 17225
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17248) (leftOrdinal := 1)
    (rightResult := 17225) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30198⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17252

namespace LeftMerge17254
def owner : Owner := ⟨.program ⟨214⟩, ⟨30199⟩⟩
def mergeEvent : Nat := 17254
def frameStart : Nat := 17174
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17027⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24803⟩⟩] } }
def rhsRaw : List Term := Proof.Events067.exact17222RawTerms
def group : MergeGroup := .relation 17253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17253) (rhsResult := 17222)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30198⟩⟩) ⟨24803⟩ 17222) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24803⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨24803⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17254

namespace LeftMerge17255
def owner : Owner := ⟨.program ⟨214⟩, ⟨30199⟩⟩
def mergeEvent : Nat := 17255
def frameStart : Nat := 17174
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17248RawTerms
def rightRaw : List Term := Proof.Events067.exact17225RawTerms
def group : MergeGroup := .operator 17248 17225
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17248) (leftOrdinal := 0)
    (rightResult := 17225) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30198⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17255

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
