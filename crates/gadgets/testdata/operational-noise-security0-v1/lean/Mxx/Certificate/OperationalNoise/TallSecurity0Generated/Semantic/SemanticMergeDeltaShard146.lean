import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge25062
def owner : Owner := ⟨.program ⟨214⟩, ⟨19759⟩⟩
def mergeEvent : Nat := 25062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩] } }
def rhsRaw : List Term := Proof.Events097.exact25058RawTerms
def group : MergeGroup := .relation 25060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25060) (rhsResult := 25058)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25059 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19756⟩⟩]⟩) (none) 25058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25062

namespace LeftMerge25063
def owner : Owner := ⟨.program ⟨214⟩, ⟨19759⟩⟩
def mergeEvent : Nat := 25063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } }
def rhsRaw : List Term := Proof.Events097.exact25058RawTerms
def group : MergeGroup := .relation 25060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25060) (rhsResult := 25058)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25059 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19756⟩⟩]⟩) (none) 25058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨23086⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25063

namespace LeftMerge25064
def owner : Owner := ⟨.program ⟨214⟩, ⟨19759⟩⟩
def mergeEvent : Nat := 25064
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events097.exact25058RawTerms
def group : MergeGroup := .relation 25060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25060) (rhsResult := 25058)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25059 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19756⟩⟩]⟩) (none) 25058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25064

namespace LeftMerge25069
def owner : Owner := ⟨.program ⟨214⟩, ⟨25159⟩⟩
def mergeEvent : Nat := 25069
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact25065RawTerms
def rightRaw : List Term := Proof.Events097.exact24879RawTerms
def group : MergeGroup := .operator 25065 24879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25065) (leftOrdinal := 2)
    (rightResult := 24879) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨23086⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25069

namespace LeftMerge25070
def owner : Owner := ⟨.program ⟨214⟩, ⟨25159⟩⟩
def mergeEvent : Nat := 25070
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact25065RawTerms
def rightRaw : List Term := Proof.Events097.exact24879RawTerms
def group : MergeGroup := .operator 25065 24879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25065) (leftOrdinal := 1)
    (rightResult := 24879) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25070

namespace LeftMerge25078
def owner : Owner := ⟨.program ⟨214⟩, ⟨28558⟩⟩
def mergeEvent : Nat := 25078
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact25072RawTerms
def rightRaw : List Term := Proof.Events096.exact24795RawTerms
def group : MergeGroup := .operator 25072 24795
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25072) (leftOrdinal := 0)
    (rightResult := 24795) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28556⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25078

namespace LeftMerge25079
def owner : Owner := ⟨.program ⟨214⟩, ⟨28558⟩⟩
def mergeEvent : Nat := 25079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact25072RawTerms
def rightRaw : List Term := Proof.Events096.exact24795RawTerms
def group : MergeGroup := .operator 25072 24795
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25072) (leftOrdinal := 1)
    (rightResult := 24795) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28556⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25079

namespace LeftMerge25081
def owner : Owner := ⟨.program ⟨214⟩, ⟨28558⟩⟩
def mergeEvent : Nat := 25081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24360⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24792RawTerms
def group : MergeGroup := .relation 25080
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25080) (rhsResult := 24792)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28556⟩⟩) ⟨24360⟩ 24792) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24360⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16274⟩⟩], [⟨.program ⟨214⟩, ⟨24360⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25081

namespace LeftMerge25095
def owner : Owner := ⟨.program ⟨214⟩, ⟨21847⟩⟩
def mergeEvent : Nat := 25095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21844⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events098.exact25089RawTerms
def group : MergeGroup := .operator 21512 25089
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 25089) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21844⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21844⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25095

namespace LeftMerge25216
def owner : Owner := ⟨.program ⟨214⟩, ⟨16351⟩⟩
def mergeEvent : Nat := 25216
def frameStart : Nat := 25150
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events098.exact25212RawTerms
def rightRaw : List Term := Proof.Events098.exact25210RawTerms
def group : MergeGroup := .operator 25212 25210
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25212) (leftOrdinal := 0)
    (rightResult := 25210) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25216

namespace LeftMerge25228
def owner : Owner := ⟨.program ⟨214⟩, ⟨28557⟩⟩
def mergeEvent : Nat := 25228
def frameStart : Nat := 25150
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩] } }
def leftRaw : List Term := Proof.Events098.exact25224RawTerms
def rightRaw : List Term := Proof.Events098.exact25201RawTerms
def group : MergeGroup := .operator 25224 25201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25224) (leftOrdinal := 0)
    (rightResult := 25201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25228

namespace LeftMerge25229
def owner : Owner := ⟨.program ⟨214⟩, ⟨28557⟩⟩
def mergeEvent : Nat := 25229
def frameStart : Nat := 25150
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩] } }
def leftRaw : List Term := Proof.Events098.exact25224RawTerms
def rightRaw : List Term := Proof.Events098.exact25201RawTerms
def group : MergeGroup := .operator 25224 25201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25224) (leftOrdinal := 1)
    (rightResult := 25201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28556⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25229

namespace LeftMerge25231
def owner : Owner := ⟨.program ⟨214⟩, ⟨28557⟩⟩
def mergeEvent : Nat := 25231
def frameStart : Nat := 25150
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24360⟩⟩] } }
def rhsRaw : List Term := Proof.Events098.exact25198RawTerms
def group : MergeGroup := .relation 25230
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25230) (rhsResult := 25198)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28556⟩⟩) ⟨24360⟩ 25198) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24360⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16274⟩⟩], [⟨.program ⟨214⟩, ⟨24360⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25231

namespace LeftMerge25239
def owner : Owner := ⟨.program ⟨214⟩, ⟨16318⟩⟩
def mergeEvent : Nat := 25239
def frameStart : Nat := 25150
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16317⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events098.exact25212RawTerms
def rightRaw : List Term := Proof.Events098.exact25235RawTerms
def group : MergeGroup := .operator 25212 25235
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25212) (leftOrdinal := 0)
    (rightResult := 25235) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16317⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25239

namespace LeftMerge25256
def owner : Owner := ⟨.program ⟨214⟩, ⟨21847⟩⟩
def mergeEvent : Nat := 25256
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }
def rhsRaw : List Term := Proof.Events098.exact25253RawTerms
def group : MergeGroup := .relation 25255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25255) (rhsResult := 25253)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21844⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25254 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21844⟩⟩]⟩) (none) 25253) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25256

namespace LeftMerge25257
def owner : Owner := ⟨.program ⟨214⟩, ⟨21847⟩⟩
def mergeEvent : Nat := 25257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩] } }
def rhsRaw : List Term := Proof.Events098.exact25253RawTerms
def group : MergeGroup := .relation 25255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25255) (rhsResult := 25253)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21844⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25254 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21844⟩⟩]⟩) (none) 25253) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28556⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25257

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
